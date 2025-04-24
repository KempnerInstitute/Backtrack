import json
import numpy as np
import matplotlib.pyplot as plt

model_result_file = '/n/netscratch/dam_lab/Lab/sqin/outputs/sos/original/sft/star3/results_train1_b4_t100_n500000_random.json_500_0_@1'
perf_result = json.load(open(model_result_file))
model_rating = np.array(perf_result["ratings"])
symbolic_rating = np.array(perf_result["true_ratings"])

# print accuracy:
model_acc = np.sum(model_rating > 0) / len(model_rating)
symbolic_acc = np.sum(symbolic_rating > 0) / len(symbolic_rating)
print(f"Model Accuracy: {model_acc*100:.2f}%")
print(f"Symbolic Accuracy: {symbolic_acc*100:.2f}%")

plt.figure()
# find instances where symbolic rating is 0
symbolic_rating_zero = np.where(symbolic_rating == 0)[0]
model_perf_when_teacher_fails = model_rating[symbolic_rating_zero]
model_perf_when_teacher_fails_acc = np.sum(model_perf_when_teacher_fails > 0) / len(model_perf_when_teacher_fails)
print(f"Model Performance when Teacher Fails: {model_perf_when_teacher_fails_acc*100:.2f}%")
# find instances where model rating is 0
model_rating_zero = np.where(model_rating == 0)[0]
teacher_perf_when_model_fails = symbolic_rating[model_rating_zero]
teacher_perf_when_model_fails_acc = np.sum(teacher_perf_when_model_fails > 0) / len(teacher_perf_when_model_fails)
print(f"Teacher Performance when Model Fails: {teacher_perf_when_model_fails_acc*100:.2f}%")