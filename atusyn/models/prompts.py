import torch

BRAIN_INFO_WEAK = {
    "csf":   "CSF often appears hyperechoic with stronger reflections near interfaces.",
    "wm":    "White matter typically shows mid-to-low echogenicity.",
    "cortex":"Cortex may show relatively enhanced edge echoes compared with adjacent tissue.",
    "tumor": "Tumor regions may present mid-to-high echogenicity with variable appearance."
}

def y1_to_label_id(y1, max_value=62):
    """
    你的 label 读入后是: 先 /max_value -> [0,1] 再 (x-0.5)/0.5 -> [-1,1]
    这里反归一化回整数类别id.
    y1: [B,C,H,W] in [-1,1]
    return: [B,H,W] long
    """
    y = y1[:, 0]  # 取第0通道即可（你label通常3通道相同）
    y01 = (y + 1.0) * 0.5
    label_id = torch.round(y01 * float(max_value)).long()
    return label_id

def build_prompt_brain_dynamic(y1, max_value=62, min_pixels=50, include_cortex=False, style="weak"):
    """
    生成类似：
    'The labelmap has 15 for CSF, 1/18 for white matter, 61 for tumor; CSF: ...; tumor: ...'
    但只包含该样本真实存在的类
    """
    label_id = y1_to_label_id(y1, max_value=max_value)  # 复用你之前的函数 [B,H,W]

    # 类定义（你给的）
    CSF_ID = 15
    WM_IDS = [1, 18]
    TUMOR_ID = 61
    CORTEX_IDS = [31, 98]  # 只有你需要时再开

    info = BRAIN_INFO_WEAK

    prompts = []
    for b in range(label_id.size(0)):
        lbl = label_id[b]
        def has(cls):
            return (lbl == cls).sum().item() > min_pixels
        def has_any(ids):
            return any(has(i) for i in ids)

        present_csf   = has(CSF_ID)
        present_wm    = has_any(WM_IDS)
        present_tumor = has(TUMOR_ID)
        present_cortex= has_any(CORTEX_IDS) if include_cortex else False

        # 第一段：labelmap has ...
        parts = []
        if present_csf:   parts.append("15 for CSF")
        if present_wm:    parts.append("1/18 for white matter")
        if present_cortex:parts.append("31/98 for cerebral cortex")
        if present_tumor: parts.append("61 for tumor")

        if len(parts) == 0:
            head = ""   # 推荐
        else:
            head = "The labelmap has " + ", ".join(parts) + "."

        # 第二段：描述（只写存在的）
        desc = []
        if present_csf:   desc.append(info["csf"])
        if present_wm:    desc.append(info["wm"])
        if present_cortex:desc.append(info["cortex"])
        if present_tumor: desc.append(info["tumor"])

        prompt = head + (" " + " ".join(desc) if len(desc) else "")
        prompts.append(prompt)

    return prompts
