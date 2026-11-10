"""Planner + executor closed loop in the LIBERO simulator (RTX 5090, one process, synchronous protocol).

Executor: ImageWAM LIBERO fp8 (FlashRT), loaded without its GPU text encoder and slimmed after the first instruction
(`colocation_qwen27b_5090/exec_slim.py`); a new instruction is encoded by the Qwen3-4B text encoder on the CPU
(bf16, encoded once per text and cached) and set through `set_prompt(context=...)`. Prompt, preprocessing, gripper
mapping, 30 wait steps, 10 executed actions per query, seed 7, per-episode action noise seed 1000 * task + episode and
700 control steps are those of `wam-inference-acceleration/experiments/e8_libero_closed_loop`.

Planner (mode "planner_eos" / "planner_fast"): Qwen3.8-27B master, FlashRT NVFP4, driving VoLoAgent's Full-loop
decision logic as `harness_replay_5090/record_calls.py` does: decompose at step 0, monitor every 80 control steps
(next / replan / continue; at most 3 replans; tool actions run no tool and count as continue), replan with the
original subgoal list. Images are the simulator's agentview (front and exterior) and wrist frames, upright, letterboxed
to the harness's camera sizes (wrist 224) and JPEG-encoded at quality 90. Synchronous protocol: during a planner call
or an instruction switch the simulator does not step.
  planner_eos  : every call decoded to EOS, whole-prompt prefill;
  planner_fast : monitor calls restore the episode's constant prefix (system prompt + BEFORE images, chunk-aligned)
                 and stop decoding once the action closes (`../monitor_call/README.md`); decompose / replan as eos.
Mode "flat": the executor alone with the task instruction (the e8 configuration). Instruction probes, executor alone:
"flat_still" with the instruction "stay still and do not move", "flat_other" with the instruction of task (task + 1) mod 10;
success is still the original task's.

Per episode: success, control steps, the event list (each planner call with its stage times, reply and decision; each
instruction switch with its CPU-encode and set_prompt times) and an annotated video (`episode_video.py`: the frame is
held for the wall time of each planner call and of each set_prompt; the CPU text encode, a setting of this demo, is
written on the frame but not held).

usage (LIBERO python, see run.sh): python closed_loop_libero.py MODE OUT_DIR TASK:EPISODE [TASK:EPISODE ...]
"""
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
WAM_EXP = os.path.join(HERE, "..", "..", "..", "wam-inference-acceleration", "experiments")
sys.path.insert(0, os.path.join(WAM_EXP, "colocation_qwen27b_5090"))
sys.path.insert(0, os.path.join(WAM_EXP, "harness_replay_5090"))
import planner_qwen27b as pq  # noqa: E402  (planner environment: checkpoint, MTP head, long-context route, FP8 KV)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from experiments.libero.libero_utils import (LIBERO_ENV_RESOLUTION, get_libero_dummy_action, get_libero_env,  # noqa: E402
                                             get_libero_image, quat2axisangle)
from libero.libero import benchmark  # noqa: E402
from PIL import Image  # noqa: E402

import flash_rt.frontends.torch.imagewam_thor as F  # noqa: E402
from flash_rt.models.imagewam.text_encoder import encode_prompts, load_real_text_encoder  # noqa: E402
from flash_rt.models.imagewam.workload import ImageWAMWorkload  # noqa: E402

import volo_calls as vc  # noqa: E402
from exec_slim import record_quantized_sources, slim_executor  # noqa: E402
from record_calls import ACTIONS, MAX_IMAGE_TOKENS, MAX_SEQ, normalize_subgoals, parse_json  # noqa: E402

from episode_video import EpisodeVideo  # noqa: E402
from planner_client import MonitorPrefix, PlannerClient, PlannerReply, action_of  # noqa: E402

PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
SUITE, SEED, NUM_STEPS_WAIT, REPLAN, MAX_STEPS = "libero_10", 7, 30, 10, 700
WRIST = 224
FPS = 20   # LIBERO control rate
MODES = ("flat", "flat_still", "flat_other", "planner_eos", "planner_fast")
STILL = "stay still and do not move"


@dataclass
class Event:
    step: int
    kind: str                 # decompose | monitor | replan | switch
    wall_ms: float
    detail: dict


@dataclass
class EpisodeRecord:
    mode: str
    task: int
    episode: int
    instruction: str
    success: bool
    control_steps: int
    executor_queries: int
    planner_ms: float         # sum of planner call wall times
    switch_ms: float          # sum of instruction switches (CPU encode + set_prompt)
    subgoals_initial: list[str]
    end_reason: str           # success | max_steps | planner_done
    video: str
    events: list[Event] = field(default_factory=list)


class Executor:
    """ImageWAM with a CPU text encoder: one context per distinct instruction text, cached."""

    def __init__(self, first_instruction: str) -> None:
        sources = record_quantized_sources(F)
        self.fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp8",
                                  ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                                  dataset_stats_path=os.environ["DATASET_STATS_PATH"])
        self.encoder = load_real_text_encoder(os.environ["QWEN3_MODEL_SPEC"], device="cpu")
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.set_instruction(first_instruction)
        self.slim = slim_executor(self.fe, sources)
        gc.collect()
        torch.cuda.empty_cache()
        self.na, self.ad = self.fe.dims["num_action"], self.fe.dims["action_dim"]

    def set_instruction(self, text: str) -> dict:
        t0 = time.perf_counter()
        cached = text in self.cache
        if not cached:
            ctx, mask = encode_prompts(*self.encoder, [PROMPT.format(task=text)], max_length=self.fe._text_max_length())
            self.cache[text] = (ctx[0], mask[0])
        t1 = time.perf_counter()
        ctx, mask = self.cache[text]
        self.fe.set_prompt(context=ctx, context_mask=mask)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return {"instruction": text, "cached": cached, "encode_cpu_ms": (t1 - t0) * 1e3,
                "set_prompt_ms": (t2 - t1) * 1e3}

    def act(self, obs: dict, gen: torch.Generator) -> list:
        noise = 0.01 * torch.randn(self.na, self.ad, generator=gen, device="cuda")
        a = np.asarray(self.fe.infer(preprocess(obs), action_noise=noise)["actions"]).copy()
        a[:, -1] = np.sign(-(a[:, -1] * 2 - 1))
        if not np.isfinite(a).all():
            raise RuntimeError("non-finite actions")
        return a[:REPLAN].tolist()


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(image)
    scale = max(width / pil.size[0], height / pil.size[1])
    r = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), resample=Image.BILINEAR)
    left, top = max((r.size[0] - width) // 2, 0), max((r.size[1] - height) // 2, 0)
    return np.asarray(r.crop((left, top, left + width, top + height)), dtype=np.uint8)


def preprocess(obs: dict) -> dict:
    imgs = get_libero_image(obs)
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"])).astype(np.float32)
    return {"view1": torch.from_numpy(center_crop_resize(imgs["image"], 224, 224).copy()),
            "view2": torch.from_numpy(center_crop_resize(imgs["wrist_image"], 224, 224).copy()),
            "proprio": state}


def harness_frames(obs: dict, step: int, out_dir: str) -> vc.HarnessFrames:
    imgs = get_libero_image(obs)
    agent, wrist = Image.fromarray(imgs["image"]), Image.fromarray(imgs["wrist_image"])
    tag = os.path.join(out_dir, f"s{step:04d}")
    return vc.HarnessFrames(front=vc._jpeg(vc._letterbox(agent, vc.FRONT_WH), tag + "_front.jpg"),
                            exterior=vc._jpeg(vc._letterbox(agent, vc.EXTERIOR_WH), tag + "_exterior.jpg"),
                            wrist=vc._jpeg(vc._letterbox(wrist, vc.WRIST_WH[WRIST]), tag + f"_wrist{WRIST}.jpg"))


def reply_detail(r: PlannerReply) -> dict:
    d = asdict(r)
    d["text"] = r.text
    return d


class ClosedLoop:
    def __init__(self, mode: str, out_dir: str) -> None:
        assert mode in MODES, mode
        self.mode, self.out_dir = mode, out_dir
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        self.executor = Executor(self.suite.get_task(0).language)
        self.planner = (PlannerClient(pq.CKPT, MAX_SEQ, MAX_IMAGE_TOKENS, pq.CHAT_KWARGS)
                        if mode.startswith("planner") else None)
        print(f"loaded: executor slim {self.executor.slim}, allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
              f"reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)

    def _plan(self, call: vc.CallSpec, stop: str = "eos", prefix: MonitorPrefix | None = None) -> PlannerReply:
        return self.planner.call(vc.to_messages(call), call.max_tokens, stop=stop, prefix=prefix)

    def episode(self, tid: int, ep: int) -> EpisodeRecord:
        task = self.suite.get_task(tid)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, SEED)
        ep_dir = os.path.join(self.out_dir, f"{self.mode}_t{tid}_e{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        video = EpisodeVideo(os.path.join(ep_dir, "episode.mp4"), FPS)
        gen = torch.Generator(device="cuda").manual_seed(1000 * tid + ep)
        env.reset()
        obs = env.set_init_state(self.suite.get_task_init_states(tid)[ep])
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        rec = EpisodeRecord(self.mode, tid, ep, desc, False, 0, 0, 0.0, 0.0, [], "max_steps",
                            os.path.relpath(video.path, self.out_dir))

        def frames() -> tuple[np.ndarray, np.ndarray]:
            imgs = get_libero_image(obs)
            return imgs["image"], imgs["wrist_image"]

        def planner_event(step: int, kind: str, r: PlannerReply, extra: dict) -> None:
            rec.events.append(Event(step, kind, r.call_ms, {**reply_detail(r), **extra}))
            rec.planner_ms += r.call_ms

        def switch(step: int, text: str, label: str) -> None:
            info = self.executor.set_instruction(text)
            ms = info["encode_cpu_ms"] + info["set_prompt_ms"]
            rec.events.append(Event(step, "switch", ms, info))
            rec.switch_ms += ms
            video.hold(*frames(), step, text, idx, len(subgoals),
                       f"{label}: instruction switch {info['set_prompt_ms'] / 1e3:.2f} s "
                       f"(+ CPU text encode {info['encode_cpu_ms'] / 1e3:.1f} s, not shown)",
                       info["set_prompt_ms"] / 1e3)

        executor_text = {"flat_still": STILL,
                         "flat_other": self.suite.get_task((tid + 1) % self.suite.n_tasks).language}.get(self.mode, desc)
        subgoals, idx = [executor_text], 0
        before = prefix = None
        original: list[str] = []
        replans = 0
        if self.planner is not None:
            before = harness_frames(obs, 0, ep_dir)
            r = self._plan(vc.decompose_call(desc, before))
            try:
                got = normalize_subgoals(parse_json(r.text).get("subgoals", []))
                ok = bool(got)
            except ValueError:
                got, ok = [], False
            subgoals = got if ok else [desc]
            planner_event(0, "decompose", r, {"parse_ok": ok, "subgoals": subgoals})
            video.hold(*frames(), 0, subgoals[0], 0, len(subgoals),
                       f"planner: decompose -> {len(subgoals)} subgoal(s) ({r.call_ms / 1e3:.1f} s)", r.call_ms / 1e3)
            if self.mode == "planner_fast":
                probe = vc.monitor_call(desc, subgoals, 0, before, before, 0)
                prefix = self.planner.build_monitor_prefix(vc.to_messages(probe), 3, probe.max_tokens)
                rec.events.append(Event(0, "monitor_prefix", prefix.build_ms, {"tokens": prefix.tokens}))
                rec.planner_ms += prefix.build_ms
                video.hold(*frames(), 0, subgoals[0], 0, len(subgoals),
                           f"planner: monitor prefix cached ({prefix.build_ms / 1e3:.2f} s)", prefix.build_ms / 1e3)
        original = list(subgoals)
        rec.subgoals_initial = list(subgoals)
        switch(0, subgoals[0], "executor")

        pending: list = []
        event_text = ""
        t, done = 0, False
        while t < MAX_STEPS:
            if self.planner is not None and t > 0 and t % vc.CHECK_INTERVAL == 0 and not pending:
                now = harness_frames(obs, t, ep_dir)
                fast = self.mode == "planner_fast"
                r = self._plan(vc.monitor_call(desc, subgoals, idx, before, now, t),
                               stop="action" if fast else "eos", prefix=prefix if fast else None)
                if fast:
                    action, ok = action_of(r.text), action_of(r.text) is not None
                else:
                    try:
                        action, ok = parse_json(r.text).get("action", "continue"), True
                    except ValueError:
                        action, ok = "continue", False
                action = action if action in ACTIONS else "continue"
                planner_event(t, "monitor", r, {"parse_ok": ok, "decision": action, "subgoal_index": idx})
                event_text = f"planner: monitor -> {action} ({r.call_ms / 1e3:.1f} s)"
                video.hold(*frames(), t, subgoals[idx], idx, len(subgoals), event_text, r.call_ms / 1e3)
                last = idx == len(subgoals) - 1
                if action == "next" and not last:
                    idx += 1
                    switch(t, subgoals[idx], "next")
                elif (action == "replan" or (action == "next" and last)) and replans < vc.MAX_REPLANS:
                    replans += 1
                    rr = self._plan(vc.replan_call(desc, original, before, now, t))
                    try:
                        data = parse_json(rr.text)
                        new, ok2, pdone = normalize_subgoals(data.get("subgoals", [])), True, bool(data.get("done"))
                    except ValueError:
                        new, ok2, pdone = [], False, False
                    accepted = bool(new) and not pdone
                    planner_event(t, "replan", rr, {"parse_ok": ok2, "done": pdone, "accepted": accepted,
                                                    "subgoals": new})
                    event_text = (f"planner: replan -> {'done' if pdone else (str(len(new)) + ' subgoal(s)')} "
                                  f"({rr.call_ms / 1e3:.1f} s)")
                    video.hold(*frames(), t, subgoals[idx], idx, len(subgoals), event_text, rr.call_ms / 1e3)
                    if accepted:
                        subgoals, idx = new, 0
                        switch(t, subgoals[0], "replan")
                    if pdone and action == "next" and last:
                        rec.end_reason = "planner_done"
                        break
            if not pending:
                pending = self.executor.act(obs, gen)
                rec.executor_queries += 1
            obs, _, done, _ = env.step(pending.pop(0))
            t += 1
            video.step(*frames(), t, subgoals[idx], idx, len(subgoals), event_text)
            if done:
                rec.end_reason = "success"
                break
        rec.success, rec.control_steps = bool(done), t
        video.close()
        env.close()
        return rec


def main() -> None:
    mode, out_dir, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    loop = ClosedLoop(mode, out_dir)
    out_path = os.path.join(out_dir, f"{mode}.json")
    records: list[EpisodeRecord] = []
    for spec in specs:
        tid, ep = (int(x) for x in spec.split(":"))
        t0 = time.time()
        rec = loop.episode(tid, ep)
        records.append(rec)
        calls = [e for e in rec.events if e.kind in ("decompose", "monitor", "replan")]
        print(f"{mode} task {tid} ep {ep}: {'S' if rec.success else 'F'} ({rec.end_reason}) steps {rec.control_steps}, "
              f"subgoals {rec.subgoals_initial}, planner calls {len(calls)} "
              f"[{' '.join(e.detail.get('decision', e.kind) for e in calls)}] {rec.planner_ms / 1e3:.1f} s, "
              f"switches {rec.switch_ms / 1e3:.1f} s, wall {time.time() - t0:.0f} s", flush=True)
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in records], f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
