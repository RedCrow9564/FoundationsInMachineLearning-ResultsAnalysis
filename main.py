import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def create_smoothness_csv(experiment_config):
    all_matrices = {}
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
        all_matrices[experiment_name] = all_data

    return all_matrices


def create_graphs(experiments_data, experiment_config):
    titles = [layer_name for layer_name in experiment_config['Layers']
              if layer_name not in experiment_config['Layers to drop']]
    index = np.arange(len(titles) + 1)
    plots = []
    a = 0
    for experiment_name, experiment_data in experiments_data.items():
        if 'original' not in experiment_name:
            mean_smoothness = [0] + np.mean(experiment_data, axis=0).tolist()
            std_smoothness = [0] + np.std(experiment_data, axis=0).tolist()
            new_plot = plt.bar(index + a, mean_smoothness, 0.4, yerr=std_smoothness)
            plots.append(new_plot)
            a += 0.4
        else:
            data = [experiment_data.tolist()[0]] + len(titles) * [0]
            new_plot = plt.bar(index, data, 0.3)
            plots.append(new_plot)

    plt.ylabel('Mean Besov smoothness')
    plt.title('CIFAR10 CNN Model Layers Smoothness')
    plt.xticks(index, ['Original Dataset'] + titles)
    #plt.ylim(0, 0.6)
    plt.legend([p[0] for p in plots], experiment_config['Name and folder'].keys())
    plt.show()


def main():
    cifar10_experiment_config = {
        'Name and folder': {
            'CIFAR10_10_epochs': os.path.join('..', 'Smoothness_results', 'CIFAR10_10_epochs'),
            'CIFAR10_20_epochs': os.path.join('..', 'Smoothness_results', 'CIFAR10_20_epochs'),
            'original_CIFAR10': os.path.join('..', 'Smoothness_results', 'original_cifar10')
        },
        'Folds count': 5,
        'Layers': ['1st conv', '2nd conv', '1st_max_pool', 'dropout', '3rd conv', '4th conv', '2nd_max_pool',
                   'dropout', 'flatten', '1st dense', 'dropout', '2nd dense-\nlogits'],
        'Layers to drop': ['flatten']
    }

    mnist_experiment_config = {
        'Name and folder': {
            'mnist_5_epochs': os.path.join('..', 'Smoothness_results', 'mnist_5_epochs'),
            'mnist_10_epochs': os.path.join('..', 'Smoothness_results', 'mnist_10_epochs'),
            'original_mnist': os.path.join('..', 'Smoothness_results', 'original_mnist')
        },
        'Folds count': 5,
        'Layers': ['1st conv', '1st max pool', '2nd conv', '2nd_max_pool', 'flatten', '1st fully connected',
                   'Dropout', '2nd fully connected-\nlogits'],
        'Layers to drop': ['flatten']
    }

    modified_mnist_config = {
        'Name and folder': {
            'modified_mnist_net': os.path.join('..', 'Smoothness_results', 'modified_mnist_net'),
            'original_mnist': os.path.join('..', 'Smoothness_results', 'original_mnist')
        },
        'Folds count': 5,
        'Layers': ['1st conv', '1st max pool', '2nd conv', '2nd_max_pool', 'flatten', '1st fully connected',
                   'Dropout', '2nd fully connected-\nlogits'],
        'Layers to drop': ['flatten']
    }

    experiment_config = cifar10_experiment_config
    all_data = create_smoothness_csv(experiment_config)
    create_graphs(all_data, experiment_config)


if __name__ == '__main__':
    main()
