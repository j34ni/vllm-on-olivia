import asyncio
import time
import httpx
import argparse


async def send_request(client, url, model, prompt, max_tokens, sem):
    async with sem:
        t0 = time.perf_counter()
        r = await client.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "enable_thinking": False,
        })
        elapsed = time.perf_counter() - t0
        data = r.json()
        out_tokens = data["usage"]["completion_tokens"]
        return elapsed, out_tokens


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    prompt = "Explain the theory of relativity in detail." + " Continue." * 50

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=300) as client:
        t_start = time.perf_counter()
        tasks = [send_request(client, url, args.model, prompt, args.max_tokens, sem)
                 for _ in range(args.num_prompts)]
        results = await asyncio.gather(*tasks)
    elapsed_total = time.perf_counter() - t_start

    latencies = [r[0] for r in results]
    total_out = sum(r[1] for r in results)
    latencies.sort()
    n = len(latencies)

    print(f"\n{'='*50}")
    print(f"Requests:          {args.num_prompts}")
    print(f"Concurrency:       {args.concurrency}")
    print(f"Max output tokens: {args.max_tokens}")
    print(f"Total time:        {elapsed_total:.2f} s")
    print(f"Output tokens:     {total_out}")
    print(f"Throughput:        {total_out/elapsed_total:.1f} tok/s")
    print(f"Req/s:             {args.num_prompts/elapsed_total:.2f}")
    print(f"Latency p50:       {latencies[int(n*0.50)]:.2f} s")
    print(f"Latency p95:       {latencies[int(n*0.95)]:.2f} s")
    print(f"Latency p99:       {latencies[int(n*0.99)]:.2f} s")
    print(f"{'='*50}\n")


asyncio.run(main())
