import fal_client
import  json
import numpy as np
import asyncio
import time
import uuid

modifications = {
    "tone": {
        "description": "Adjusting the tone of the conversation can help make the video more engaging. A more exciting, humorous, or dramatic tone can make the conversation feel more dynamic.",
        "examples": {
        "slightly": "The speaker remains mostly calm, but adds slight enthusiasm in certain parts. 'I think this idea could be really important... perhaps the most important.'",
        "moderate": "The speaker sounds more enthusiastic but not over the top. 'This idea could change everything! It's absolutely revolutionary!'",
        "extreme": "The speaker adopts a high-energy, almost hyperactive tone, making the conversation feel like a hype-building speech. 'This idea will shake the world! EVERYTHING WILL CHANGE!'",
        "theatric": "The speaker uses dramatic pauses and over-the-top emotion, almost like performing for an audience. 'THIS... is the moment we’ve all been waiting for! The entire world will be altered!'",
        "neutral": "The speaker sticks to a neutral tone throughout, not overly excited or downplayed. 'This idea is interesting, and we should look at it closely.'",
        "intense": "The speaker’s voice is deeply serious and passionate, emphasizing urgency and importance. 'If we don’t act NOW, we risk losing EVERYTHING!'"
        }
    },
    "ordering_of_conversation": {
        "description": "Rearranging the sequence of the conversation can dramatically alter the pacing and impact. Putting the most attention-grabbing or controversial statements at the beginning or end can make the video more captivating.",
        "examples": {
        "slightly": "Place a mild provocative statement at the start. 'You might not believe it, but time might not be as we think it is.'",
        "moderate": "Switching parts of the conversation so the speakers build up to a bigger reveal. 'It sounds impossible... But we’ve discovered time isn’t linear.'",
        "extreme": "Switch the order so that the most exciting part comes immediately. 'Did you know that time could be manipulated? Here's how it works!'",
        "theatric": "Start with a highly dramatic or shocking statement and build anticipation. 'Time itself bends and warps! Let me show you how that’s possible…!'",
        "neutral": "Keep the order of conversation the same as the original, without adding any abrupt shifts. 'Time is a complex concept that we're trying to understand. It’s fascinating!'",
        "intense": "The conversation begins with an urgent statement that grabs attention instantly. 'If we don’t understand how time works, everything could collapse! Here’s why.'"
        }
    },
    "hook_placement": {
        "description": "Strategically placing a hook (something provocative or interesting) early in the video to keep viewers from scrolling. The hook should engage their curiosity or make them want to keep watching.",
        "examples": {
        "slightly": "Introduce an intriguing but mild statement early on. 'You won’t believe how time is not what you think.'",
        "moderate": "Use a more curious hook at the start. 'Ever thought about time bending? Let me explain why it might be possible.'",
        "extreme": "Drop a bombshell hook early on to capture immediate interest. 'What if I told you time doesn’t exist as you think?'",
        "theatric": "Start with a cliffhanger hook that promises something mind-blowing. 'What you’re about to hear might just change everything you know about the universe!'",
        "neutral": "The hook is subtle and builds over time. 'Today, we’re going to talk about time and its strange properties. Stay tuned to learn more.'",
        "intense": "The hook is direct and urgent, demanding attention. 'Stop scrolling! What if time was never linear at all? You need to hear this!'"
        }
    },
    "pacing": {
        "description": "Adjusting the pacing of the conversation helps maintain viewer interest. Speeding up the dialogue can make it feel more urgent, while slowing it down can build suspense or emphasis.",
        "examples": {
        "slightly": "The conversation flows naturally with slight pauses after key statements. 'So, time... is it as we think? Well, let’s dive deeper.'",
        "moderate": "Speeding up sections to make the content feel faster-paced. 'It’s not linear. In fact, it's much more complex! Here's why.'",
        "extreme": "Rapid-fire pacing with little to no pauses, creating a sense of urgency and excitement. 'Time bends, time stretches, time is not what we think. Here’s proof!'",
        "theatric": "Slower pacing, with deliberate pauses for effect, creating tension. 'Time... is not as we perceive. It’s far deeper than we know... let me explain.'",
        "neutral": "Natural pacing where there’s a mix of fast and slow. 'It’s a big idea, but let’s start small. Time... is more than just a ticking clock.'",
        "intense": "Very quick pacing that delivers the conversation with urgency and emphasis. 'We don’t have time to waste, so here’s what you need to know now!'"
        }
    },
    "introduction_and_outro": {
        "description": "Modifying how you introduce and close the video can influence how memorable and impactful the conversation is.",
        "examples": {
        "slightly": "A short intro with a basic intro phrase. 'Hey, I’m going to talk about time. Stay with me.'",
        "moderate": "An engaging intro with a question. 'Ever wonder if time is really as it seems? Let’s talk about it.'",
        "extreme": "An attention-grabbing intro that teases the conversation. 'Get ready to have your mind blown by this one idea about time!'",
        "theatric": "A strong, dramatic introduction and outro. 'This is the most important conversation you’ll hear today. Don’t miss it.'",
        "neutral": "A straightforward intro and outro. 'Let’s discuss the nature of time and what we know about it.'",
        "intense": "A bold and impactful intro, leading straight into the core content. 'This video could change your life. Time is not what you think it is!'",
        "loop": "The end of the video ends with a statement that is continued at the beggining of the video. At the start 'Knowing this has changed my life...' right at the end of the video '...the strength of gravitational pull.And thats why...'"
        }
    }
}
    

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            pass

def fill_prompt_clip(script,clip_length, n_clips):

    return f"""
You are an editor for a famous internet star. They want you to grab their scripts and break them up into segments to make interesting videos. These clips must contain enough grabbing information for users to be hooked to the video, you can take any part of the script and combine them as you wish into one script. You should prioritize your selection to be in the scipt provided, but you can link non-contiguous parts together with sentences of your own. The clips you make must be at most {clip_length} words each, you can make as many as you would like but they must be different. If there is an unknown speaker in your clip, mark them as Speaker_n where is the number of the unidentified speaker. Finally, if your clips have more than one speaker, you should provide a clip where each speaker speaks at most once in the interaction.

Provide your answer as an array of arrays of dictionaries. The dictionary should break up the script by speaker, so they should have two keys "speaker" and "text". If there is only one speaker there should only be one dictionary with all the sentences of this speaker in one fluid text, if there is more than one speaker, there shall be as many dictionaries in this array as speakers and must be broken down by order of speech, but any one speaker can onle speak once. Return the arrays and nothing else, do not explain your answer at all or provide any information other than the clip arrays. Make sure you close the last array.

Be careful to not create any new text or discussion beyond what is provided, you must base your answer solely on the script provided. Create at most {n_clips} clips from the provided script.

The script you must use as the source of clips is this:
{script}
    """

def fill_prompt_AB_test(script,direction = ""):
    
    d = {
    'transcription' : '<enter edited script>',
    'speaker_voice_descriptions': {'<speaker_name':'<description>'},
    'modifications' : '<modifications_used>',
    'short_modifications': '<summarisation_of_modifications>',
    'title': '<snappy_title>',
    'description': '<short_description_with_hasthags>',
    
    }
    
    l = f"""
You will receive a script in the form of an array of dictionaries, with the keys "speaker" and "text". Your job is to make this as entertaining as possible whilst still being truthful to the script provided. Edit the scripts provided to maximise the probability that a user will stop to listen to the interaction or monologue given. You can do one of the following things to make it more entertaining:

{modifications}


{direction}
Your response shall be a dictionary with two keys: 
{d}

Provide your edited text in the same exact format as you are given it in the "transcription" key. Also give a short description of how you imagine the speakers voice as well, make sure you add identifiable information like sex, depth of voice, and emotions you want to convey. It should be a short and sweet description. Provide a list of modifications applied to the initial script in the appropiate section in your reponse. The next field should be a very short description of these modifications. Also, provide a title for the clip which is snappy and reflects the feeling of the clip, add all the emojis you want to this as well. A longer description should be created with a similar intention with the addition of hastags which can be used to identify this viral clip. Provide no additional information but this data structure. Do not add indications that are visual, focus on the dialogue and make sure to get the emotions in the text they are going to read out loud only, do not add any onomatopoeia or indications of actions, or activity, just focus what would be spoken and nothing else. 

The original script is:
{script}
"""
    return l

def fill_prompt_direction(clip, old_modifications, reward_signal, n_directions = 2):

    return f"""
    You are a genius influencer, and you are trying to make your videos as viral as possible. You have to help your editor understand what they can improve in their edited scripts to make them more likely that it is seen. Your editor can only do the following things to make the videos more entertaining:

    {modifications}

    The original script is:

    {clip}

    They previously implemented these modifications:

    {old_modifications}

    The uploaded model had the following summary of statistics:

    {reward_signal}

    You must provide advice for your new clip, so that it can create a more grabbing video, which improves these statistics.

    Return your advice as a text file no more than 100 words. Be concise and precise. Remember that the editor only works with audio so do not recommend any onomatopoeia or visual queues, only focus on the conversations/words. Provide {n_directions} comentaries, and return them as a array of strings.
"""

async def prompt_llm(prompt, model='anthropic/claude-3.5-sonnet', max_attempts=100):
    """
    Returns a set of valid clips asynchronously.
    """
    attempts = 0
    while attempts < max_attempts:
        # Wrap the synchronous call in asyncio.to_thread
        clips = await asyncio.to_thread(
            fal_client.subscribe,
            "fal-ai/any-llm",
            arguments={
                "model": model,         # Specify the LLM model
                "prompt": prompt,       # User input
                "system_prompt": "Make sure you make the output a json loadable string",
                "seed": str(np.random.randint(0, int(1e6)))
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
            
        try:
            jsoned = json.loads(clips['output'])
        except Exception as e:
            try:
                jsoned = json.loads(clips['output'] + "]")
                
            except Exception as e:
                attempts += 1
                continue
        return jsoned
    return clips['output']

async def get_AB_candidates(candidate_clips, n_ab_clips=2, direction=''):
    """
    Given candidate clips, launch AB test generation concurrently.
    Yields each AB candidate as soon as it's available.
    """
    tasks = []
    for clip in candidate_clips:
        for i in range(n_ab_clips):
            prompt_text = fill_prompt_AB_test(clip, direction)
            task = asyncio.create_task(prompt_llm(prompt=prompt_text, max_attempts=10))
            tasks.append(task)

    # Yield results as soon as each task completes.
    for future in asyncio.as_completed(tasks):
        ab_clip = await future
        yield ab_clip

async def get_AB_direction(candidate_clip, context, applied_modifications,  n_directions=2,):
    """
    Given candidate clips, launch AB test generation concurrently.
    Yields each AB candidate as soon as it's available.
    """ 
    prompt_text = fill_prompt_direction(candidate_clip, context, applied_modifications, n_directions)
    return await prompt_llm(prompt=prompt_text, max_attempts=10)
    
async def improve_AB_candidates(candidate_transcription_clip, context, applied_modifications, n_directions=2):
    """
    Given a candidate transcription clip, a context, and applied modifications,
    concurrently generate improved AB candidates (one per direction) and yield
    each improved candidate as soon as it is available.
    """
    # Get directions from the AB direction prompt.
    # Expecting get_AB_direction to return a list of direction strings.
    directions = await get_AB_direction(candidate_transcription_clip, context, applied_modifications, n_directions)

    # Define a helper that collects a single improved candidate for a given direction.
    async def generate_for_direction(direction):
        # Wrap the single candidate in a list so that get_AB_candidates works correctly.
        async for candidate in get_AB_candidates([candidate_transcription_clip], n_ab_clips=1, direction=direction):
            return candidate
        return None  # In case nothing is yielded

    # Create a task for each direction.
    tasks = [asyncio.create_task(generate_for_direction(direction)) for direction in directions]

    # Yield each improved candidate as soon as it is available.
    for future in asyncio.as_completed(tasks):
        candidate = await future
        if candidate is not None:
            yield candidate
     
async def get_initial_AB_candidates(transcription, clip_length=300, n_distinct_clips=2, n_ab_clips=2):
    """
    Generates candidate clips then concurrently generates AB candidates.
    Returns an asynchronous generator yielding each AB candidate as it is ready.
    """
    candidate_clips = await prompt_llm(
        prompt=fill_prompt_clip(transcription, clip_length=clip_length, n_clips=n_distinct_clips),
        max_attempts=10
    )
    # candidate_clips should be a list of valid clips.
    async for ab_clip in get_AB_candidates(candidate_clips, n_ab_clips=n_ab_clips):
        ab_clip_reformat = {}
        for k,v in ab_clip.items():
            if k in ['transcription','speaker_voice_descriptions']:
                ab_clip_reformat[k] = v
            else:
                if 'params' not in ab_clip_reformat.keys():
                    ab_clip_reformat['params'] = {}
                ab_clip_reformat['params'][k] = v


        ab_clip_reformat['params']['id'] = str(uuid.uuid4())
            
        yield ab_clip_reformat



def main(transcription):
    """Synchronous main function that reads AB candidates as they come."""
    candidates = []
    now = time.time()
    
    # Create or get an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Get the asynchronous generator from get_initial_AB_candidates.
    agen = get_initial_AB_candidates(transcription, clip_length=100)
    
    while True:
        try:
            # Retrieve the next candidate from the async generator.
            candidate = loop.run_until_complete(agen.__anext__())
            print("Received an AB candidate:", candidate)
            candidates.append(candidate)
            print("Elapsed time:", time.time() - now)
            now = time.time()
        except StopAsyncIteration:
            # No more candidates.
            break
    
    return candidates

if __name__ == "__main__":
    # Replace this with your actual transcription or script.
    transcription = "This is a creative treat, imagine any content you would like and make a clip out of it."
    
    # Call the synchronous main function.
    all_candidates = main(transcription)
    print("Final candidates:", all_candidates)