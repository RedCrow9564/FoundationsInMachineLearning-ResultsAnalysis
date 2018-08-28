import pandas as pd
import numpy as np
import os


def create_smoothness_csv(experiment_config):
    all_matrices = []
    layers_count = len(experiment_config['Layers'])
    folds_count = experiment_config['Folds count']

    for experiment_name, results_path in experiment_config['Name and folder'].items():
        all_data = []
        if 'original' in experiment_name:
            layers_count = 1

        for layer_index in range(0, layers_count):
            if not experiment_config['Layers'][layer_index] in experiment_config['Layers to drop']:
                layer_results = []
                for fold_index in range(folds_count):
                    if 'original' in experiment_name:
                        smoothness_file_path = os.path.join(results_path, str(fold_index), 'alphaMterm.txt')
                    else:
                        smoothness_file_path = os.path.join(results_path, 'layer_{0}'.format(layer_index),
                                                           str(fold_index), 'alphaMterm.txt')
                    with open(smoothness_file_path) as smoothness_file:
                        result = float(smoothness_file.readlines()[0].rstrip('\n'))
                        layer_results.append(result)
                # mean_smoothness = np.mean(layer_results)
                # layer_results.append(mean_smoothness)
                all_data.append(layer_results)

        all_data = np.array(all_data).transpose()
        df = pd.DataFrame(all_data)
        if 'original' in experiment_name:
            df.columns = [experiment_name]
        else:
            df.columns = [layer_name for layer_name in experiment_config['Layers']
                          if layer_name not in experiment_config['Layers to drop']]
        df.to_csv(experiment_name + '.csv')
        all_matrices.append(all_data)

    return all_matrices


def main():
    experiment_config = {
        'Name and folder': {
            'MNIST_5_epochs': os.path.join('..', 'Smoothness_results', 'MNIST_5_epochs'),
            'MNIST_10_epochs': os.path.join('..', 'Smoothness_results', 'MNIST_10_epochs'),
            #'original_MNIST': os.path.join('..', 'Smoothness_results', 'original_MNIST')
        },
        'Folds count': 5,
        'Layers': ['1st conv', '1st_max_pool', '2nd conv', '2nd_max_pool', 'flatten', '1st_fully_connected (no dropout)',
                   'dropout', '2nd_fully_connected-logits'],
        'Layers to drop': ['flatten', 'dropout']
    }

    all_matrices = create_smoothness_csv(experiment_config)
    pass


if __name__ == '__main__':
    main()
