import pandas as pd

def evaluate_results(experiment: str, filename: str):
    print(f'Evaluating "{experiment}"...\n')
    df = pd.read_csv(filename)

    selected_answers = [x for x in df['Selected Answer']]
    correct_answers = [x for x in df['Correct Answer']]

    zipped_results = zip(selected_answers, correct_answers)

    evaluated_results = 0
    answered_correctly = 0

    for selected_answer, correct_answer in zipped_results:
        correct = selected_answer == correct_answer

        print(f"Selected: {selected_answer} & Correct: {correct_answer} | {correct}")

        evaluated_results += 1
        answered_correctly += 1 if correct else 0

    print("\n-------------------------------- Final Results --------------------------------")

    print(f"Answered Questions: {evaluated_results}")
    print(f"Correctly Answered Questions: {answered_correctly}")
    print(f"Incorrect Answered Questions: {evaluated_results - answered_correctly}")

    print(f"|#> Reached Accuracy: {answered_correctly / evaluated_results}")

def main():
    evaluate_results('Unbiased - DeepSeekR1', 'results/truthful_qa_results_unbiased.csv')


if __name__ == '__main__':
    main()