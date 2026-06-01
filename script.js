const templates = {
    zno: {
        name: "ZnO 纳米颗粒",
        method: "水热合成法",
        params: [
            ["前驱体", "醋酸锌"],
            ["煅烧温度", "500℃"]
        ]
    },
    tio2: {
        name: "TiO₂ 光催化剂",
        method: "溶胶-凝胶法",
        params: [
            ["前驱体", "钛酸四丁酯"]
        ]
    },
    ceo2: {
        name: "CeO₂ 掺杂材料",
        method: "水热法",
        params: [
            ["掺杂量", "0.5%"]
        ]
    }
};

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("zno").onclick = () => load("zno");
    document.getElementById("tio2").onclick = () => load("tio2");
    document.getElementById("ceo2").onclick = () => load("ceo2");
    document.getElementById("genBtn").onclick = generate;
    addRow();
});

function load(key) {
    const t = templates[key];
    document.getElementById("sample_name").value = t.name;
    document.getElementById("method").value = t.method;
    document.getElementById("params").innerHTML = "";
    t.params.forEach(p => addRow(p[0], p[1]));
}

function addRow(k = "", v = "") {
    const div = document.createElement("div");
    div.className = "param-row";
    div.innerHTML = `
        <input class="k" value="${k}">
        <input class="v" value="${v}">
        <button onclick="this.parentElement.remove()">×</button>
    `;
    document.getElementById("params").appendChild(div);
}

async function generate() {
    const btn = document.getElementById("genBtn");
    btn.textContent = "⏳ AI 生成中...";
    btn.disabled = true;

    const params = {
        "样品名称": document.getElementById("sample_name").value,
        "制备方法": document.getElementById("method").value
    };

    document.querySelectorAll(".param-row").forEach(row => {
        const k = row.querySelector(".k").value;
        const v = row.querySelector(".v").value;
        if (k) params[k] = v;
    });

    try {
        // 修改为相对路径 /api/generate
        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });

        const data = await res.json();

        if (data.success) {
            // 修改为相对路径 /result
            window.location.href = "/result";
        } else {
            alert("生成失败：" + (data.error || "未知错误"));
        }
    } catch (err) {
        alert("无法连接后端服务，请确保 app.py 已启动！");
        console.error(err);
    }

    btn.textContent = "🚀 生成对比图谱";
    btn.disabled = false;
}
