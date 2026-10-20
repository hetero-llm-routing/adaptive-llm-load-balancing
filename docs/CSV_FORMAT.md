# Results file format (agreed by the team)

The router writes one row per request.

| Column | Meaning |
| --- | --- |
| run_id | Name of the experiment run (from config.yaml) |
| strategy | Strategy used |
| seed | Random seed of the load tester (sent in the `x-seed` header) |
| request_id | Unique id of the request |
| gpu | Node that served the request |
| prompt_len | Prompt length in tokens (estimated) |
| output_len | Number of generated tokens |
| arrival_ts | Arrival time at the router (Unix seconds) |
| ttft_ms | Time to first token, in milliseconds |
| tpot_ms | Time per output token after the first, in milliseconds |
| e2e_ms | Total time of the request, in milliseconds |
| energy_j | Energy of the request in joules (empty until the energy logger is added) |
