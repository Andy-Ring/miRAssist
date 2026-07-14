# Installing the miRAssist Skill in Claude — Step by Step

This guide is written for bench scientists. It assumes **no computer background** — if you
can use a web browser and Claude, you can do this. It takes about **5 minutes**, once.

A "skill" is just an add-on that teaches Claude how to do a specific job. The **miRAssist
skill** teaches Claude to look up and rank microRNA–target interactions from our evidence
database and explain the results. Once it's installed, you simply ask Claude questions in
plain English like *"Which miRNAs regulate PTEN?"*

You only install it once. After that, it's always available in your Claude account.

---

## Before you start

You need:

1. **The Claude app** (the desktop app or the website, https://claude.ai), and you're
   signed in. Any paid Claude plan works.
2. **The skill file**, called `mirassist_skill.zip` (you'll download it in Step 1).

That's it. You do **not** need to install Python, create any accounts, or type any code.
Claude handles all of that for you behind the scenes.

---

## Step 1 — Download the skill file

1. In your web browser, go to the miRAssist page on GitHub:
   **https://github.com/Andy-Ring/miRAssist/releases**
2. Under the latest release, find the file named **`mirassist_skill.zip`** in the list of
   files (the "Assets" list).
3. **Click it.** Your browser will download it, usually to your **Downloads** folder.
4. That's all — you don't need to open or unzip it. Just remember it's in your Downloads
   folder.

> 💡 If you were sent the `mirassist_skill.zip` file directly (by email or a shared drive),
> just save it somewhere you can find it, like your Downloads folder, and skip to Step 2.

---

## Step 2 — Turn on "code execution" in Claude (one-time)

The skill works by having Claude run a small program for you, so this setting must be on.

1. Open the **Claude** app (or claude.ai in your browser).
2. Click your **name or profile picture** (bottom-left corner) and choose **Settings**.
3. Look for a section called **Capabilities** (sometimes shown under "Features").
4. Find **Code execution** (it may also be called "Analysis tool" or "Code") and switch it
   **On**.

You only ever have to do this once.

---

## Step 3 — Add the miRAssist skill

1. Still in Claude, open the **Settings / Customize** menu and click **Skills**.
2. Click the **`+`** (plus) button, then choose **Create skill**.
3. When it asks for a file, select the **`mirassist_skill.zip`** you downloaded in Step 1
   (look in your Downloads folder).
4. Claude will add it. You'll see **miRAssist** appear in your list of skills, with a
   switch next to it. Make sure the switch is **On**.

The skill is now installed. 🎉

---

## Step 4 — Ask your first question

1. Start a **new chat** in Claude.
2. Type a question in plain English, for example:

   > Which miRNAs regulate PTEN?

3. Press Enter and wait.

**The very first question takes a little longer — about a minute.** Behind the scenes,
Claude is doing one-time setup: downloading the miRAssist evidence database (about 106 MB)
and getting ready. This only happens once; every question after that is quick.

You'll know it's working when Claude replies with a **ranked list of candidates** and a
short explanation of the evidence behind each one.

---

## Example questions you can ask

You can ask in normal language — no special format needed:

- *"What genes does miR-21 target?"*
- *"Which miRNAs are predicted to regulate PTEN in breast cancer?"*
- *"Show me the top targets of miR-155 involved in apoptosis."*
- *"I overexpressed miR-34a and saw increased apoptosis — what direct targets could explain
  that?"*
- *"Give me high-confidence, novel targets of miR-10b in prostate cancer."*

Tips:
- Mention a **cancer type** (breast, colon, or prostate) to bring in cancer-specific
  evidence.
- Mention a **process** (apoptosis, proliferation, EMT, invasion, migration) to focus on
  genes in that pathway.
- Ask for **"high-confidence"** results to require multiple lines of evidence, or
  **"novel"** results to focus on targets that aren't already experimentally confirmed.

---

## What the answer means

For each candidate, Claude reports only what the database actually contains — it will not
make up numbers. You'll typically see:

- The **ranked candidates** (top of the list = strongest overall evidence).
- The **types of evidence** behind each one (for example: sequence match, binding data,
  conservation, cancer expression).
- A note on **how it was ranked** (the miRAssist machine-learning score).

These are **prioritized candidates to guide your work**, not proven interactions — always
plan to confirm the ones you care about at the bench.

---

## Troubleshooting

**"Nothing happens / Claude says it can't run code."**
Code execution isn't on. Go back to **Step 2** and switch it on, then start a new chat.

**"It couldn't download the database" (or a download error).**
Your network may be blocking the download. Try again on a normal internet connection; if it
keeps failing at work, ask your IT department to allow downloads from `github.com`. You can
also let us know and we'll help.

**"Claude didn't use the skill."**
Make sure the miRAssist switch is **On** in your Skills list (Step 3). You can also nudge it
by starting your message with *"Using miRAssist, ..."*.

**"The first answer is slow."**
That's expected the very first time (the one-time database download). Later questions are
fast.

---

## Updating the skill later

If a newer version of miRAssist is released, just download the new `mirassist_skill.zip`
and repeat **Step 3** — adding it again will replace the old version. The database updates
itself automatically.

---

## Prefer not to install anything?

You don't have to use the skill at all — the same tool is available as a **website** you can
just open and use:
**https://andy-ring-mirassist.share.connect.posit.cloud/**

The skill is simply a convenience for people who already work inside Claude.

---

*Sources for the install steps: Claude Help Center —*
*[Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude),*
*[How to create custom skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills).*
