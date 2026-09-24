"""
main.py — 入口
用法：
  python main.py "你的研究问题"
  python main.py          （交互式输入）
"""

import sys
import os

# 把 searchAgent/ 本身加入 sys.path，使 agent.py 中的绝对导入可以找到 config、tools、context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Deep Research Agent")
        print("输入 'quit' 退出\n")
        query = input("请输入研究问题：").strip()

    if not query or query.lower() == "quit":
        print("已退出")
        sys.exit(0)

    result = agent.run(query)

    print("\n" + "=" * 60)
    print("研究完成")
    if result:
        print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
