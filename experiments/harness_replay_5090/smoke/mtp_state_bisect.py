"""Which earlier call degrades MTP acceptance later in the same process: probe = one recorded monitor call at
tail 0, K 3 (spec cycles / accepted drafts); run it once in a fresh process, then after each (synthetic prompt, tail
config) block of mtp_tail_model's grid in grid order."""
import sys

import mtp_tail_model as mt


def main(calls: str) -> None:
    bench = mt.TailBench()
    synth, real = mt.prompts(bench, [calls])
    probe, _ = bench.reference(real[1], 512)

    def run_probe(after: str) -> None:
        c = bench.cell(probe, "0", "-", "3")
        print(f"probe after {after}: cycles {c.spec_cycles}, accepted {c.spec_accepted}, same {c.same_as_reference}",
              flush=True)

    run_probe("start")
    for spec in synth:
        spec, _ = bench.reference(spec, spec.max_new_tokens)
        for tail, fc in mt.TAIL_CONFIGS:
            for k in mt.KS:
                bench.cell(spec, tail, fc, k)
            run_probe(f"{spec.name} tail={tail}/{fc}")


if __name__ == "__main__":
    main(sys.argv[1])
