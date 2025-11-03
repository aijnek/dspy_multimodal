# Setup
import dspy
from PIL import Image
from create_dataset import create_count_dataset, split_dataset

# カスタムメトリクス関数
def count_exact_match(example, pred, trace=None):
    """
    人物カウントの完全一致メトリクス

    Args:
        example: 正解ラベルを含むExampleオブジェクト
        pred: プログラムの予測結果
        trace: 中間ステップ（オプショナル）

    Returns:
        bool or float: 一致した場合True/1.0、不一致の場合False/0.0
    """
    return example.number_of_people == pred.number_of_people

# 単一画像のテスト
def test_single_image():
    """単一画像で人物カウントをテスト"""
    print("=" * 60)
    print("単一画像テスト")
    print("=" * 60)

    image = Image.open("images/count/0/zero.jpg")
    image.thumbnail((1024, 1024), Image.LANCZOS)

    lm = dspy.LM('openai/gemma3:4b', api_base='http://localhost:11434/v1', api_key='not_needed')
    dspy.configure(lm=lm)

    p = dspy.Predict("image: dspy.Image -> number_of_people: int")(image=dspy.Image.from_PIL(image))
    print(f"予測された人数: {p.number_of_people}")
    print()

# データセットでの評価
def evaluate_dataset():
    """データセット全体で評価を実行"""
    print("=" * 60)
    print("データセット評価")
    print("=" * 60)

    # データセットの作成
    print("\nデータセットを読み込み中...")
    dataset = create_count_dataset()
    trainset, devset, testset = split_dataset(dataset)

    print(f"訓練セット: {len(trainset)}個")
    print(f"開発セット: {len(devset)}個")
    print(f"テストセット: {len(testset)}個")

    # LMの設定
    lm = dspy.LM('openai/gemma3:4b', api_base='http://localhost:11434/v1', api_key='not_needed')
    dspy.configure(lm=lm)

    # プログラムの定義
    program = dspy.Predict("image: dspy.Image -> number_of_people: int")

    # 開発セットで評価
    print("\n開発セットで評価中...")
    print("-" * 60)

    correct = 0
    total = len(devset)

    for i, example in enumerate(devset, 1):
        # 予測
        pred = program(**example.inputs())

        # count_exact_matchを使用して評価
        # メトリクスは example と pred を比較
        is_correct = count_exact_match(example, pred)

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"{status} [{i}/{total}] 正解: {example.number_of_people}, 予測: {pred.number_of_people}")

    accuracy = correct / total if total > 0 else 0
    print("-" * 60)
    print(f"\n正解数: {correct}/{total}")
    print(f"精度: {accuracy:.2%}")
    print()

if __name__ == "__main__":
    # 単一画像テスト
    test_single_image()

    # データセット評価
    evaluate_dataset()

    # dspy.inspect_history()