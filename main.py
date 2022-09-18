from nextcord import Interaction, SlashOption, Embed
from nextcord.ext import commands
import transformers
import random
import airtable as at
import glob
import pickle

models = glob.glob("models/*")

if "models/unmasker.aimodel" in models:
    unmasker = pickle.load("models/unmasker.aimodel")
else:
    unmasker = transformers.pipeline("fill-mask", model="bert-base-uncased")
    unmaskerfile = open("models/unmasker.aimodel", "w+b")
    pickle.dump(unmasker, unmaskerfile)

if "models/summarizer.aimodel" in models:
    summarizer = pickle.load("models/summarizer.aimodel")
else:
    summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")
    summarizerfile = open("models/summarizer.aimodel", "w+b")
    pickle.dump(summarizer, summarizerfile)

if "models/textgenerator.aimodel" in models:
    textgenerator = pickle.load("models/textgenerator.aimodel")
else:
    textgenerator = transformers.pipeline("text-generation", model="gpt2")
    textgeneratorfile = open("models/textgenerator.aimodel", "w+b")
    pickle.dump(textgenerator, textgeneratorfile)

db = at.Airtable('appWlNWcMJ1Yj3qCQ', 'keylqCeR5Mr62P71A')

bot = commands.Bot()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


@bot.slash_command(
    name="report",
    description="Report a bug with the bot or model",
    dm_permission=True
)
async def report(
        ctx: Interaction,
        title=SlashOption(
            name="title",
            description="The title of your bug report"
        ),
        description=SlashOption(
            name="description",
            description="The description of your bug report"
        ),
        reproducesteps=SlashOption(
            name="reproducesteps",
            description="The steps to reproduce your issue"
        )
):
    db.create("Bugs",
              {"Title": title, "Description": description, "Discord User": ctx.user.name,
               "Reproduce Steps": reproducesteps})
    await ctx.send("Reported your issue", ephemeral=True)


@bot.slash_command(
    name="request",
    description="Request a feature for the bot",
    dm_permission=True
)
async def request(
        ctx: Interaction,
        title=SlashOption(
            name="title",
            description="The title of your request"
        ),
        description=SlashOption(
            name="description",
            description="The description of your request"
        )
):
    db.create("Feature",
              {"Title": title, "Description": description, "Discord User": ctx.user.name})
    await ctx.send("Requested", ephemeral=True)


@bot.slash_command(
    name="requestmodel",
    description="Request a model for the bot",
    dm_permission=True
)
async def requestmodel(
        ctx: Interaction,
        title=SlashOption(
            name="title",
            description="The title of your request"
        ),
        description=SlashOption(
            name="description",
            description="The description of your request"
        )
):
    db.create("Model",
              {"Title": title, "Description": description, "Discord User": ctx.user.name})
    await ctx.send("Requested", ephemeral=True)


@bot.slash_command(
    name="fill-mask",
    description="Finds the best string to fill the mask marked with [MASK]",
    dm_permission=True
)
async def fillmask(
        ctx: Interaction,
        mask: str = SlashOption(
            name="text",
            description="The text you want to use. Make sure it has [MASK] in it."
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(with_message=True)
    if mask.endswith("[MASK]"):
        await ctx.send("Please add a period `.` to the end of your sentence", ephemeral=True)
    else:
        from transformers.pipelines import PipelineException
        try:
            response = unmasker(mask)[0]
            mostCommonResponse = response["token_str"]
            mostCommonResponseScore = response["score"]
            fullStr = mask.replace("[MASK]", mostCommonResponse)
            scorePercent = str(round(mostCommonResponseScore * 100)) + "%"
            embed = Embed(title="Fill-mask generation complete:")
            embed.add_field(name="Original:", value=f"{mask}", inline=False)
            embed.add_field(name="AI generated:", value=f"\"{fullStr}\"", inline=True)
            embed.add_field(name="Score:", value=f"{scorePercent}", inline=True)
            await ctx.send(embed=embed)
        except PipelineException:
            await ctx.send("The [MASK] token could not be found in the text", ephemeral=True)


@bot.slash_command(
    name="summarize",
    description="Provides a brief summary of a long string of text",
    dm_permission=True
)
async def summarize(
        ctx: Interaction,
        text: str = SlashOption(
            name="text",
            description="The text you want to be summarized"
        ),
        maxLength: int = SlashOption(
            name="maximumlength",
            description="The maximum length of the summarized text"
        ),
        minLength: int = SlashOption(
            name="minimumlength",
            description="The minimum length of the summarized text"
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    summarized = summarizer(text, max_length=maxLength, min_length=minLength, do_sample=False)
    summarized = summarized[0]["summary_text"]
    showSummarized = True
    showText = True
    if len(summarized) > 1024:
        showSummarized = False
    if len(text) > 1024:
        showText = False
    embed = Embed(title="Summarization complete:")
    if showText:
        embed.add_field(name="Original:", value=f"\"{text}\"", inline=False)
    else:
        embed.add_field(name="Original:", value=f"Not shown due to length greater than 1024 characters.", inline=False)
    embed.add_field(name="Minimum Length:", value=f"{minLength}", inline=True)
    embed.add_field(name="Maximum Length:", value=f"{maxLength}", inline=True)
    if showSummarized:
        embed.add_field(name="AI Generated:", value=f"\"{summarized}\"", inline=False)
    else:
        embed.add_field(name="AI Generated:", value=f"Not shown due to length greater than 1024 characters. DMed to you instead.", inline=False)
        await ctx.user.send(f"Summarized: \"{summarized}\"")
    await ctx.send(embed=embed)


@bot.slash_command(
    name="generate",
    description="Generates text to extend the text you input",
    dm_permission=True
)
async def generate(
        ctx: Interaction,
        text: str = SlashOption(
            name="text",
            description="The text to extend"
        ),
        maxLength: int = SlashOption(
            name="maxlength",
            description="The maximum length of the generated text"
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    transformers.set_seed(random.randint(0, 99))
    generated = textgenerator(text, max_length=maxLength, num_return_sequences=1)[0]["generated_text"]
    generated = generated.replace(f"{text}", "")
    await ctx.send(
        f"""```ansi
{text}[2;40m[2;34m[2;41m[2;45m[2;30m[2;37m[2;47m[2;30m{generated}[0m[2;37m[2;47m[0m[2;37m[2;45m[0m[2;30m[2;45m[0m[2;34m[2;45m[0m[2;34m[2;41m[0m[2;34m[2;40m[0m[2;40m[0m
```"""
    )


bot.run("MTAxNTc0OTMxMTcyMTY1NjQ4MQ.Gv7aO4.G7qDOOuOIOdiP7RYMkgomUwpNsL52vW32KL21M")
