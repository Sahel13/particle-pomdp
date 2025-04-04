# Experiments

Usage:
```bash
./run.sh <algorithm> <environment>
```
Outputs will be saved in `./logs/<algorithm>_<environment>_<timestamp>`.

Available algorithms:
1. nested nsmc (ours)
2. nested ncsmc (ours)
2. slac
3. dsmc (dual smc)


Available environments:
1. pendulum
2. cartpole
3. target-sensing
4. light-dark-1d
5. light-dark-2d

## TODO
- [ ] Add an `evaluate` function for dual smc.
