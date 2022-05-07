import pandas as pd

name_to_csv_path = {
    'ID-CIFAR-10-MadrysResnet': '/home/uriel/research/pnml_ood_detection/outputs/odin_MadrysResnet_cifar10_20220504_223433/performance.csv',
    'ID-CIFAR-10-ResNet18': '/home/uriel/research/pnml_ood_detection/outputs/odin_resnet18_cifar10_20220504_214413/performance.csv',
    'ID-CIFAR-10-ResNet34': '/home/uriel/research/pnml_ood_detection/outputs'
                            '/odin_resnet34_cifar10_20220504_214758/performance.csv',
    'ID-CIFAR-100-MadrysResnet':
        '/home/uriel/research/pnml_ood_detection/outputs/odin_MadrysResnet_cifar100_20220504_214252/performance.csv',
    'ID-CIFAR-100-ResNet18': '/home/uriel/research/pnml_ood_detection/outputs/odin_resnet18_cifar100_20220504_222626/performance.csv',
    'ID-CIFAR-100-ResNet34': '/home/uriel/research/pnml_ood_detection/outputs/odin_resnet34_cifar100_20220504_225930/performance.csv'
}
for experiment_name in name_to_csv_path:
    path_to_csv = name_to_csv_path[experiment_name]
    df = pd.read_csv(path_to_csv)
    new_df = df[['model', 'ood_name', 'AUROC_odin', 'AUROC_pnml']]
    print(experiment_name)
    print(new_df.to_latex(index=False))
    print("\n \n \n")
    print("~" * 50)
    print("\n \n \n")