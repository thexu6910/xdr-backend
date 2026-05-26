from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 强制使用无界面后端，解决 PythonAnywhere 报错
import matplotlib.pyplot as plt
import time
import re
import logging
from zhipuai import ZhipuAI
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ==================== 智谱AI客户端初始化 ====================
api_key = os.getenv("ZHIPU_API_KEY")
if not api_key:
    raise ValueError("未找到ZHIPU_API_KEY，请在.env文件中配置")
client = ZhipuAI(api_key=api_key)

# ==================== 标准XRD库 ====================
STANDARD_XRD = {
    "ZnO": {
        "2theta": [31.77, 34.42, 36.25, 47.54, 56.60, 62.86, 67.95],
        "intensity": [100, 60, 80, 40, 30, 25, 20],
        "name": "ZnO 标准图谱"
    },
    "TiO2": {
        "2theta": [25.28, 37.80, 48.05, 54.72, 62.70, 68.76],
        "intensity": [100, 30, 45, 15, 20, 18],
        "name": "TiO2 标准图谱"
    },
    "CeO2": {
        "2theta": [28.55, 33.08, 47.48, 56.33, 59.09, 69.41],
        "intensity": [100, 45, 50, 25, 20, 15],
        "name": "CeO2 标准图谱"
    }
}

app = Flask(__name__)
CORS(app)

# 全局保存结果
last_result = {
    "image_standard": "",
    "image_ai": "",
    "params": {}
}


# ==================== 1. 首页路由 ====================
@app.route('/')
def index():
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'xrd-frontend')
    return send_from_directory(frontend_dir, 'index.html')


# ==================== 2. 结果页路由 ====================
@app.route('/result')
def result():
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'xrd-frontend')
    return send_from_directory(frontend_dir, 'result.html')


# ==================== 3. 静态文件路由 ====================
@app.route('/<filename>')
def serve_frontend_static(filename):
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'xrd-frontend')
    if filename in ['style.css', 'script.js']:
        return send_from_directory(frontend_dir, filename)
    return send_from_directory('static', filename)


# ==================== 4. 生成接口 ====================
@app.route("/api/generate", methods=["POST"])
def api_generate():
    global last_result
    params = request.json

    if not params:
        return jsonify({"success": False, "error": "请求参数不能为空"})

    try:
        img_standard, img_ai = generate_dual_image(params)
        last_result = {
            "image_standard": img_standard,
            "image_ai": img_ai,
            "params": params
        }
        return jsonify({"success": True, "message": "生成成功"})
    except ValueError as e:
        return jsonify({"success": False, "error": f"数据错误: {str(e)}"})
    except Exception as e:
        return jsonify({"success": False, "error": f"生成失败: {str(e)}"})


# ==================== 5. 获取结果接口 ====================
@app.route("/api/get-result")
def api_get_result():
    return jsonify(last_result)


# ==================== 生成双图函数 ====================
def generate_dual_image(params):
    sample = params.get("样品名称", "")
    method = params.get("制备方法", "")

    if not sample:
        raise ValueError("样品名称不能为空")

    # 获取标准数据
    standard = None
    for key in STANDARD_XRD:
        if key in sample:
            standard = STANDARD_XRD[key]
            break
    if not standard:
        standard = STANDARD_XRD["ZnO"]

    # 调用智谱AI生成XRD数据
    prompt = f"""你是材料XRD专家，请根据以下信息生成XRD图谱数据。

材料：{sample}
制备方法：{method}

要求：
1. 只返回JSON格式，不要任何解释文字
2. 必须包含2theta和intensity两个字段
3. 2theta范围在20-80度之间
4. intensity范围在0-100之间
5. 至少包含5个衍射峰

返回格式示例：
{{"2theta": [31.77, 34.42, 36.25, 47.54, 56.60], "intensity": [100, 60, 80, 40, 30]}}"""

    try:
        logger.info(f"调用AI生成XRD数据，样品: {sample}, 方法: {method}")

        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"AI返回内容长度: {len(content)}")

        # 清理 Markdown 代码块 ```json ... ```
        if "```" in content:
            parts = content.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{"):
                    content = p
                    break
                if p.lower().startswith("json"):
                    content = p[4:].strip()
                    break

        # 提取JSON部分（找到第一个 { 和最后一个 }）
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]

        # 修复常见JSON格式问题
        content = content.replace("'", '"')
        content = re.sub(r'(\w+):', r'"\1":', content)

        ai_data = json.loads(content)
        logger.info(f"成功解析JSON，包含 {len(ai_data.get('2theta', []))} 个衍射峰")

        # 验证数据格式
        if "2theta" not in ai_data or "intensity" not in ai_data:
            raise ValueError(f"AI返回的数据缺少必要字段")

        if len(ai_data["2theta"]) == 0 or len(ai_data["intensity"]) == 0:
            raise ValueError("AI返回的数据为空")

        if not isinstance(ai_data["2theta"], list) or not isinstance(ai_data["intensity"], list):
            raise ValueError("2theta和intensity必须是数组格式")

        if len(ai_data["2theta"]) != len(ai_data["intensity"]):
            raise ValueError("2theta和intensity数组长度不一致")

    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise ValueError(f"AI返回的JSON格式错误: {str(e)}")
    except AttributeError:
        raise ValueError("AI响应格式异常")
    except Exception as e:
        logger.error(f"AI调用失败: {str(e)}")
        raise

    # 生成平滑曲线
    x = np.linspace(10, 80, 1000)
    y_std = np.zeros_like(x)
    for t, i in zip(standard["2theta"], standard["intensity"]):
        y_std += np.exp(-((x - t) / 0.25)**2) * i

    y_ai = np.zeros_like(x)
    for t, i in zip(ai_data["2theta"], ai_data["intensity"]):
        y_ai += np.exp(-((x - t) / 0.25)**2) * i

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 保存图片
    os.makedirs("static", exist_ok=True)
    timestamp = int(time.time())
    
    # 生成标准图谱
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5), dpi=100)
    ax1.plot(x, y_std, label=standard["name"], color="#007bff", linewidth=2)
    ax1.set_title("标准XRD图谱", fontsize=14, fontweight='bold')
    ax1.set_xlabel("2θ (°)", fontsize=12)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)
    plt.suptitle(f"样品: {sample} | 方法: {method}", fontsize=12, y=0.98)
    plt.tight_layout()
    
    img_name_std = f"xrd_standard_{timestamp}.png"
    img_path_std = f"static/{img_name_std}"
    plt.savefig(img_path_std, bbox_inches="tight", dpi=150)
    plt.close()
    
    # 生成AI图谱
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5), dpi=100)
    ax2.plot(x, y_ai, label="AI生成", color="#dc3545", linewidth=2)
    ax2.set_title("AI生成XRD图谱", fontsize=14, fontweight='bold')
    ax2.set_xlabel("2θ (°)", fontsize=12)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)
    plt.suptitle(f"样品: {sample} | 方法: {method}", fontsize=12, y=0.98)
    plt.tight_layout()
    
    img_name_ai = f"xrd_ai_{timestamp}.png"
    img_path_ai = f"static/{img_name_ai}"
    plt.savefig(img_path_ai, bbox_inches="tight", dpi=150)
    plt.close()

    return f"/static/{img_name_std}", f"/static/{img_name_ai}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5186, debug=True)
