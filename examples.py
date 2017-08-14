from phmm import PHMM, PHMM_d

"""
Some example uses of the PHMM and PHMM_d classes. Note that it might take some
finessing to find optimal starting values for the HMM parameters.

"""

def phmm_test():
    # Parameters
    theta = [[0.5, 0.5],
             [0.5, 0.5]]
    delta = [1.0, 0.0]
    lambdas = [1.0, 5.4]
    # Random parameters
    theta_2 = [[0.3, 0.7],
             [0.7, 0.3]]
    delta_2 = [1.0, 0.0]
    lambdas_2 = [0.5, 6.7]
    # Initialization
    # Initialization
    h_1 = PHMM(delta, theta, lambdas)
    seqs, states = zip(*[h_1.gen_seq() for _ in range(20)])
    h_2 = PHMM(delta_2, theta_2, lambdas_2)
    h_2.baum_welch(seqs)
    print(h_2.transition_matrix())
    print(h_2.lambdas)
    v_2 = list(map(h_2.viterbi, seqs))
    print(list(zip(states[0], v_2[0])))
    print(h_2.log_likelihood(seqs))

def phmm_d_test():
    # Parameters
    theta = [[0.5, 0.5],
             [0.5, 0.5]]
    delta = [1.0, 0.0]
    lambdas = [[1.0, 5.4]] * 10
    # Random parameters
    theta_2 = [[0.3, 0.7],
             [0.7, 0.3]]
    delta_2 = [1.0, 0.0]
    lambdas_2 = [[0.5, 6.7]] * 10
    # Initialization
    h_1 = PHMM_d(delta, theta, lambdas)
    seqs, states = zip(*[h_1.gen_seq(i) for i in range(10)])
    h_2 = PHMM_d(delta_2, theta_2, lambdas_2)
    h_2.baum_welch(seqs)
    print(h_2.transition_matrix())
    print(h_2.lambdas)
    v_2 = list(map(h_2.viterbi, range(len(seqs)), seqs))
    print(list(zip(states[0], v_2[0])))
    print(h_2.log_likelihood(seqs))

def main():
    phmm_test()
    # phmm_d_test()


if __name__ == "__main__":
    main()
