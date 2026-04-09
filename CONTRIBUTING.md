# Contributing to LATRACE / 参与 LATRACE 贡献

Thanks for your interest in improving LATRACE. We welcome bug reports, documentation updates, tests, performance improvements, and feature PRs.

感谢你愿意参与 LATRACE 的建设。我们欢迎 bug 修复、文档补充、测试增强、性能优化和功能型 Pull Request。

## Before you start / 提交前

- Read the README and the relevant docs for the area you want to change.
- If you are opening an issue, please use the most relevant issue template.
- Search existing issues and pull requests to avoid duplicate work.
- For large changes, open an issue first so we can align on scope before you spend time implementing it.
- Keep changes small and focused whenever possible.

- 先阅读 README 和与你要修改的功能相关的文档。
- 如果你要提 issue，请优先选择最相关的 issue 模板。
- 先搜索已有 issue 和 PR，避免重复劳动。
- 如果是较大的改动，建议先开 issue 对齐范围，再开始实现。
- 尽量让每个 PR 保持小而聚焦。

## Local setup / 本地开发

The project uses `uv` for dependency management.

```bash
uv sync
uv run pytest
```

If you need the full service stack, use Docker Compose:

```bash
docker compose up --build
```

项目使用 `uv` 管理依赖。

```bash
uv sync
uv run pytest
```

如果你需要启动完整服务栈，可以使用 Docker Compose：

```bash
docker compose up --build
```

## What to contribute / 欢迎贡献的内容

- Bug fixes
- Tests and regression coverage
- Documentation improvements
- Refactors that make the code easier to understand or maintain
- New features that fit the project roadmap

- Bug 修复
- 测试补充和回归覆盖
- 文档改进
- 让代码更易读、更易维护的重构
- 符合路线图的新功能

## Workflow / 提交流程

1. Create or pick an issue that describes the problem or feature.
2. Create a branch for your work.
3. Make the smallest useful change you can.
4. Add or update tests when behavior changes.
5. Update docs if the user-facing behavior changes.
6. Open a pull request and describe what changed and how you verified it.

1. 先创建或认领一个 issue，说明问题或功能需求。
2. 为你的工作创建独立分支。
3. 尽量只做最小但有价值的改动。
4. 行为变化时，请补充或更新测试。
5. 用户可见行为变化时，请同步更新文档。
6. 提交 Pull Request，并说明改了什么、怎么验证的。

## Branch and commit style / 分支与提交建议

- Use short, descriptive branch names like `fix-typo-readme` or `feat-retrieval-cache`.
- Write commit messages that explain the intent of the change.
- Prefer one logical change per PR.

- 分支名建议简短且清晰，例如 `fix-readme-typo` 或 `feat-retrieval-cache`。
- 提交信息尽量说明这次改动的目的。
- 一个 PR 尽量只承载一类逻辑变更。

## Pull request checklist / PR 检查清单

- The change is scoped and easy to review.
- Tests pass locally.
- New behavior is covered by tests when practical.
- Documentation has been updated if needed.
- The PR description explains the why, the what, and the verification steps.

- 变更范围清晰，便于审查。
- 本地测试通过。
- 如果可行，新行为已补充测试覆盖。
- 如有必要，文档已同步更新。
- PR 描述里写清楚了改动原因、改动内容和验证方式。

## Review process / 审查流程

All pull requests will be reviewed by the project maintainer before merge. I may request changes, ask for tests, or suggest a smaller follow-up if that makes the review safer and faster.

所有 Pull Request 都会在合并前由项目维护者审查。我可能会提出修改建议、要求补充测试，或者建议拆成更小的后续 PR，这样能让合并更稳、更快。

## Need help? / 需要帮助？

If you are unsure where to start, open an issue or contact: [zx19970301@gmail.com](mailto:zx19970301@gmail.com)

如果你不确定从哪里开始，可以先开 issue，或者联系邮箱：[zx19970301@gmail.com](mailto:zx19970301@gmail.com)
