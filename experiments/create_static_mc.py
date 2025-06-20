from environments import MarkovChain

env1 = MarkovChain.setup_random(num_states=21, seed=1337)
env1.save_to_npz("results/chain_21.npz")
print("Saved static Markov chains to 'results/m'")

env2 = MarkovChain.setup_random(num_states=51, seed=4242)
env2.save_to_npz("results/chain_51.npz")

print("Saved static Markov chains to 'results/'")
