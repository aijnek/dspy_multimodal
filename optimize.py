# Setup
import os
from dotenv import load_dotenv

import dspy
from dspy import GEPA
from dspy.evaluate import Evaluate
from PIL import Image
from create_dataset import create_count_dataset, split_dataset

load_dotenv()

# カスタムメトリクス関数
def count_exact_match_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    人物カウントの完全一致メトリクス(feedbackつき)

    Args:
        example: 正解ラベルを含むExampleオブジェクト
        pred: プログラムの予測結果
        trace: 中間ステップ（オプショナル）

    Returns:
        correctness: 一致した場合True、不一致の場合False
        feedback_text: テキストフィードバック
    """
    correctness = (example.number_of_people == pred.number_of_people)
    feedback_text = f"""Correct answer is {example.number_of_people}. 
    Your answer is {pred.number_of_people}. Your reasoning is: {pred.reasoning}. 
    """
    if correctness:
        feedback_text += 'Your answer is correct! '
    else:
        feedback_text += 'Your answer is incorrect. Think about how you could have correct answer. '
    if pred.number_of_people >= 10:
        feedback_text += 'If number of peoplne in the image >= 10, answer 10.'

    return dspy.Prediction(score=correctness, feedback=feedback_text)


# データセットでの評価
def optimize():
    """最適化を実行"""
    print("=" * 60)
    print("最適化")
    print("=" * 60)

    # データセットの作成
    print("\nデータセットを読み込み中...")
    dataset = create_count_dataset()
    trainset, valset, testset = split_dataset(dataset)

    print(f"訓練セット: {len(trainset)}個")
    print(f"評価セット: {len(valset)}個")
    print(f"テストセット: {len(testset)}個")

    # LMの設定
    lm = dspy.LM('openai/gemma3:27b', api_base='http://localhost:11434/v1', api_key='not_needed')
    dspy.configure(lm=lm)
    reflection_lm = dspy.LM('anthropic/claude-haiku-4-5-20251001', api_key=os.getenv('ANTHROPIC_API_KEY'))

    # optimizerの設定
    optimizer = GEPA(
        metric=count_exact_match_with_feedback,
        auto="light",
        reflection_lm=reflection_lm,
        num_threads=1,
        reflection_minibatch_size=3, # token limitを超える場合は減らす
    )

    # プログラムの定義
    program = dspy.ChainOfThought("image: dspy.Image -> number_of_people: int")

    # 最適化の実行
    optimized_program = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset,
    )
    output_path = 'optimized.json'
    optimized_program.save(output_path)

if __name__ == "__main__":
    optimize()
