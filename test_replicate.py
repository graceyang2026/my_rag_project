import os
import replicate

if "REPLICATE_API_TOKEN" not in os.environ:
    print(" [警告] REPLICATE_API_TOKEN 环境变量未设置")
else:
    print(" [成功] 正在测试调用 Replicate API...")
    try:
        output = replicate.run(
            "meta/meta-llama-3-8b-instruct",
            input={"prompt": "Hello! Give me a one-sentence greeting."}
        )
        print("\n [成功] Replicate API 调用成功，返回结果：")
        print("".join(output))
    except Exception as e:
        print(f"\n [错误] 调用失败: {e}")
