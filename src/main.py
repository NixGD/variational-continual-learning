import argparse
import experiments.discriminative
import experiments.generative


# experiments from the VCL paper that can be carried out
EXP_OPTIONS = {
    'disc_p_mnist': experiments.discriminative.permuted_mnist,
    'disc_s_mnist': experiments.discriminative.split_mnist,
    'disc_s_n_mnist': experiments.discriminative.split_not_mnist,
    'gen_mnist': experiments.generative.generate_mnist,
    'gen_not_mnist': experiments.generative.generate_not_mnist
}


def main(experiment='all'):
    # run all experiments
    if experiment == 'all':
        for exp in list(EXP_OPTIONS.keys()):
            print("Running", exp)
            EXP_OPTIONS[exp]()
    # select specific experiment to run
    else:
        EXP_OPTIONS[experiment]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment', help='Experiment to be run, can be one of: ' + str(list(EXP_OPTIONS.keys())))
    args = parser.parse_args()
    main(args.experiment)
