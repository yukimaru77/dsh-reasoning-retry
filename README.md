# dsh-reasoning-retry

DeepSeek などで、返答が**推論だけで出力上限に達し、文章もツール呼び出しも出せずに止まる**
（「Output token limit reached」）問題に、**指示の出し方**で対応する
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) プラグインです。
推論を止めたり、推論の強さや出力上限を下げたりはしません。

## 1. 起きにくくする（常時の指示）

対象のモデル（既定: provider/model に `deepseek` を含むもの）のセッションに、次の指示を足します。

- 1回の返答の出力上限（例: 32768トークン）と、推論もそこに含まれること
- 1回で全部を解こうとせず、次の一手を決めたら動く（ツール呼び出しか文章）。残りの検討は次のステップへ
- 長い検討は、途中の結論や短い計画を文章（または todo_write やファイル）に出しながら進める

## 2. 起きたら、推論を引き継いでやり直す

推論だけで上限に達して止まったターンを見つけると、次の内容で同じセッションに再送します。

- 前回は推論だけで上限に達し、その作業は保存されなかったこと
- **前回の推論の末尾（既定4000文字）**。やり直しではなく、そこから続ける
- 必要なだけ考えてよいが、今回は返答の中で進捗を見える形にする（今の結論・計画を文章で出すか、次の一手をツールで実行）。残りは次のステップへ
- 同じ言い回しを繰り返していた場合は、その点を具体的に示し、決めるか（ツールで確かめて）先へ進むよう促す
- 2回目は「作業をもっと小さく分け、残りの手順を短い計画として書いてから、最初の1つだけ行う」を加える

2回やり直しても止まる場合は、goal を止めて（blocked）利用者に知らせます。
やり直しが進んだら、止まっていた goal を再開します。

## 設定（`cordis.patch.yml`）

| 項目 | 既定 | 意味 |
| --- | --- | --- |
| `maxRetries` | 2 | やり直しの回数 |
| `guidanceModels` | `deepseek` | 常時の指示を足すモデル（provider/model に対する正規表現。空文字で無効） |
| `reasoningTailChars` | 4000 | やり直しに含める前回の推論の末尾の文字数 |
| `repetition` | ngram 8, minCount 5 | 繰り返しの検出 |
| `resumeGoalAfterRetry` / `blockGoalAfterRetries` / `notifyOnGiveUp` | true | goal の再開・停止と通知 |

## 導入

```sh
dsh plugin --profile web add "file:$PWD"
```

## テスト

```sh
node scripts/link-dsh.mjs ~/.local/share/dsh/versions/0.2.1-rc.2/node_modules   # または DSH のソースcheckout
npm test
```

## ライセンス

MIT
