# Clip-Generator-AB-Test

## Documentation

[Project's Google Doc](https://docs.google.com/document/d/1bOnIdPGhDHB_Nl-u1p_v7Dn0aMATKQhH4lao0ZH_BjA/edit?tab=t.0)

[Hackathon's Docs](https://docs.google.com/document/d/1fsxiceA97FyWSx8uWC0ulPF6cY7Biusy0dgbcHdMzlk/edit?pli=1&tab=t.0)

## Environment Management with Pipenv

### Setup Instructions for Collaborative Environment

#### 1. Install Pipenv

First, ensure that you have Pipenv installed on your local machine. If you don’t have it, you can install it via pip:

```bash
pip install pipenv
```

#### 2. Create a Virtual Environment (No need as its already done)

Navigate to your project directory and initialize a new virtual environment using Pipenv:

```bash
cd /path/to/your/project
pipenv install
```

This will create a `Pipfile` in your project directory. This file is used to manage your project's dependencies.

#### 3. Install Project Dependencies

To install the packages your project needs, use the following command:

```bash
pipenv install <package_name>
```

Replace `<package_name>` with the library or package you want to install (e.g., `numpy`, `pandas`, etc.).

#### 4. Generate `requirements.txt`

If you need a `requirements.txt` file (for example, to share with other environments), you can generate it using:

```bash
pipenv lock -r > requirements.txt
```

This will create a `requirements.txt` file with the exact versions of all your dependencies, ensuring consistency across all environments.

### 6. Activate the Virtual Environment

To activate the virtual environment and work within it, use:

```bash
pipenv shell
```

Your command prompt will change to indicate that the virtual environment is now active.

### 7. Deactivate the Virtual Environment

Once you're done working in the virtual environment, you can deactivate it by simply typing:

```bash
exit
```

## Task distribution

Podcast Clip Generator + Automated AB testing

Task A.1 = Extract Transcription from input content
Task A.2 = Generate Audio from M Variations
Task B.1 = Generate text clips from input transcription
Task B.2 = Generate text variants from text clips
Task C.1 = Identify places to put music + what type of music + what specific track
Task C.2 = Identify places to put sound effects + generate prompt for elevenlabs for sound effect
Task C.3 = Identify main topics + Search internet for images
Task C.4 = Put everything together
Task D.1 = From a set of input videos, post them on tiktok
Task D.2 = Analyse posted videos: overall views, like, comments (sentiment analysis)
Task D.3 = Generate a report with the results and tell the user which one he should put more money on for example

Task E = Generate the UI for all of this

A = Iñaki
B = Mike
C = Pascu + Iñaki
D = jesus

# Docs

Elevenlabs - https://docingest.com/docs/elevenlabs.io

Lovable Dev - https://docingest.com/docs/docs.lovable.dev

Fal.ai - https://docingest.com/docs/docs.fal.ai

Clerk - https://docingest.com/docs/clerk.com

Mistral - https://docingest.com/docs/docs.mistral.ai

Picaos - https://docingest.com/docs/docs.picaos.com

# Upload videos to youtube

https://videoflo.app/tutorial/
