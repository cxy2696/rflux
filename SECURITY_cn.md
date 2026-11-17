# 安全策略

[English](SECURITY.md) · **中文**

## 支持版本

| 版本 | 支持范围 |
| --- | --- |
| 0.7.x | 安全修复 |
| < 0.7 | 尽力而为 |

## 报告漏洞

请使用本仓库的 GitHub **private vulnerability reporting / security advisory**
流程。未修复漏洞不要开公开 issue。

报告中请包含：

- 受影响版本和操作系统；
- 最小复现或 PoC；
- 问题影响 C ABI、pybind11 扩展、数据层还是 wheel 打包；
- 任何疑似凭据泄露。

## 敏感数据

rflux 在 import 时不要求任何数据源凭据。数据适配器会把行情缓存在调用方指定
的本地文件中；请按数据授权要求保护这些文件。

## 范围

C ABI、pybind11 扩展、wheel 打包、数据适配器和原生内存边界在范围内。第三方
数据源 SDK、行情授权和交易亏损不在本安全策略范围内。
