"""
人物カウント用のExampleデータセットを作成するスクリプト
images/count/の下にあるディレクトリ名が正解ラベル（人数）を表す
"""
import dspy
from PIL import Image
from pathlib import Path
import random

def create_count_dataset():
    """
    images/count/の下にある画像からExampleデータセットを作成

    Returns:
        list: dspy.Exampleオブジェクトのリスト
    """
    dataset = []
    count_dir = Path("images/count")

    # 0から10までの各ディレクトリを走査
    for label_dir in sorted(count_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        # ディレクトリ名が正解ラベル（人数）
        try:
            num_people = int(label_dir.name)
        except ValueError:
            print(f"警告: {label_dir.name} は数値ではないためスキップします")
            continue

        # ディレクトリ内の画像ファイルを収集
        image_extensions = ['.jpg', '.jpeg', '.png']
        for image_path in label_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                try:
                    # 画像を読み込んでサムネイル化
                    img = Image.open(image_path)
                    img.thumbnail((1024, 1024), Image.LANCZOS)

                    # dspy.Exampleを作成
                    example = dspy.Example(
                        image=dspy.Image.from_PIL(img),
                        number_of_people=num_people
                    ).with_inputs("image")

                    dataset.append(example)
                    print(f"追加: {image_path.name} (人数: {num_people})")

                except Exception as e:
                    print(f"エラー: {image_path} の読み込みに失敗 - {e}")

    return dataset

def split_dataset(dataset, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """
    データセットを訓練/開発/テストセットに分割

    Args:
        dataset: 全データセット
        train_ratio: 訓練セットの割合
        dev_ratio: 開発セットの割合
        test_ratio: テストセットの割合

    Returns:
        tuple: (trainset, devset, testset)
    """
    # シャッフル
    random.seed(42)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    # 分割点を計算
    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    trainset = shuffled[:train_end]
    devset = shuffled[train_end:dev_end]
    testset = shuffled[dev_end:]

    return trainset, devset, testset

if __name__ == "__main__":
    print("=" * 60)
    print("人物カウント用Exampleデータセットの作成")
    print("=" * 60)

    # データセット作成
    print("\n画像を読み込んでデータセットを作成中...")
    dataset = create_count_dataset()

    print(f"\n合計 {len(dataset)} 個のExampleを作成しました")

    # データセットの統計情報
    print("\n--- データセットの内訳 ---")
    label_counts = {}
    for example in dataset:
        label = example.number_of_people
        label_counts[label] = label_counts.get(label, 0) + 1

    for label in sorted(label_counts.keys()):
        print(f"人数 {label}: {label_counts[label]}枚")

    # 訓練/開発/テストセットに分割
    print("\nデータセットを分割中...")
    trainset, devset, testset = split_dataset(dataset)

    print(f"訓練セット: {len(trainset)}個")
    print(f"開発セット: {len(devset)}個")
    print(f"テストセット: {len(testset)}個")

    # サンプルの表示
    print("\n--- サンプルExample ---")
    if dataset:
        sample = dataset[0]
        print(f"入力: {sample.inputs()}")
        print(f"ラベル: {sample.labels()}")
        print(f"人数: {sample.number_of_people}")

    print("\n完了!")
    print("\n使用例:")
    print("  from create_dataset import create_count_dataset, split_dataset")
    print("  dataset = create_count_dataset()")
    print("  trainset, devset, testset = split_dataset(dataset)")
