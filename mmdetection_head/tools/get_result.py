import argparse
import pickle as pk

def get_dict():
    output = dict()
    output[1] = 0
    output[2] = 0
    output[3] = 0
    output[4] = 0
    output[5] = 0
    return output

def get_acc(corrects, cnt):
    if cnt==0:
        acc=-1
    else:
        acc = corrects/cnt
    return acc

def main(args):

    with open(args.result_path, "rb") as f:
        results = pk.load(f)

    # 1 shape
    corrects_shape = get_dict()
    cnt_shape = get_dict()
    accs_shape = get_dict()

    # 2 color/skin
    corrects_color = get_dict()
    cnt_color = get_dict()
    accs_color = get_dict()

    # 3 eye
    corrects_eye = get_dict()
    cnt_eye = get_dict()
    accs_eye = get_dict()

    # 4 cut
    corrects_cut = get_dict()
    cnt_cut = get_dict()
    accs_cut = get_dict()

    # 5 gill/선별?
    corrects_gill = get_dict()
    cnt_gill = get_dict()
    accs_gill = get_dict()

    # 6 dryness
    corrects_dryness = get_dict()
    cnt_dryness = get_dict()
    accs_dryness = get_dict()

    # 7 offness
    corrects_offness = get_dict()
    cnt_offness = get_dict()
    accs_offness = get_dict()

    # 8 final freshness
    corrects_final = get_dict()
    cnt_final = get_dict()
    accs_final = get_dict()

    for result in results:
        # 1
        shape_scores = result['pred_instances']['clf_logits'][:,0:]
        shape_pred = shape_scores.argmax().item()
        shape_target = result['clf_score'][0]
        cnt_shape[shape_target]+=1
        if shape_pred+1 == shape_target:
            corrects_final[shape_target] += 1

        # 2
        color_scores = result['pred_instances']['clf_logits'][:,1:]
        color_pred = color_scores.argmax().item()
        color_target = result['clf_score'][1]
        cnt_color[color_target]+=1
        if color_pred+1 == color_target:
            corrects_final[color_target] += 1

        # 3
        eye_scores = result['pred_instances']['clf_logits'][:,2:]
        eye_pred = eye_scores.argmax().item()
        eye_target = result['clf_score'][2]
        cnt_eye[eye_target]+=1
        if eye_pred+1 == eye_target:
            corrects_final[eye_target] += 1
        # 4
        cut_scores = result['pred_instances']['clf_logits'][:,3:]
        cut_pred = cut_scores.argmax().item()
        cut_target = result['clf_score'][3]
        cnt_cut[cut_target]+=1
        if cut_pred+1 == cut_target:
            corrects_final[cut_target] += 1

        # 5
        gill_scores = result['pred_instances']['clf_logits'][:,4:]
        gill_pred = gill_scores.argmax().item()
        gill_target = result['clf_score'][4]
        cnt_gill[gill_target]+=1
        if gill_pred+1 == gill_target:
            corrects_final[gill_target] += 1

        # 6
        dryness_scores = result['pred_instances']['clf_logits'][:,5:]
        dryness_pred = dryness_scores.argmax().item()
        dryness_target = result['clf_score'][5]
        cnt_dryness[dryness_target]+=1
        if dryness_pred+1 == dryness_target:
            corrects_final[dryness_target] += 1

        # 7
        offness_scores = result['pred_instances']['clf_logits'][:,6:]
        offness_pred = offness_scores.argmax().item()
        offness_target = result['clf_score'][6]
        cnt_final[offness_target]+=1
        if offness_pred+1 == offness_target:
            corrects_final[offness_target] += 1

        # 8
        freshness_score = result['pred_instances']['clf_logits'][:,-1:]
        freshness_pred = freshness_score.argmax().item()
        freshness_target = result['clf_score'][-1]
        cnt_final[freshness_target]+=1
        if freshness_pred+1 == freshness_target:
            corrects_final[freshness_target] += 1

    for i in range(1, 5):
        accs_shape[i] = get_acc(corrects_shape[i], cnt_shape[i])
        accs_color[i] = get_acc(corrects_color[i], cnt_color[i])
        accs_eye[i] = get_acc(corrects_eye[i], cnt_eye[i])
        accs_cut[i] = get_acc(corrects_cut[i], cnt_cut[i])
        accs_gill[i] = get_acc(corrects_gill[i], cnt_gill[i])
        accs_dryness[i] = get_acc(corrects_dryness[i], cnt_dryness[i])
        accs_offness[i] = get_acc(corrects_offness[i], cnt_offness[i])
        accs_final[i] = get_acc(corrects_final[i], cnt_final[i])
        print(f"shape {i}: {accs_shape[i]}")
        print(f"color {i}: {accs_color[i]}")
        print(f"eye {i}: {accs_eye[i]}")
        print(f"cut {i}: {accs_cut[i]}")
        print(f"gill {i}: {accs_gill[i]}")
        print(f"dryness {i}: {accs_dryness[i]}")
        print(f"offness {i}: {accs_offness[i]}")
        print(f"final freshness {i}: {accs_final[i]}")

    with open(args.save_path, "wb") as f:
        pk.dump(corrects_shape, f)
        pk.dump(cnt_shape, f)
        pk.dump(corrects_color, f)
        pk.dump(cnt_color, f)
        pk.dump(corrects_eye, f)
        pk.dump(cnt_eye, f)
        pk.dump(corrects_cut, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()
    main(args)


