from nextcord import Interaction, SlashOption, Embed, ui, ButtonStyle
from nextcord.ext import commands
import transformers
import random
import airtable as at
import glob
import pickle
from detoxify import Detoxify

models = glob.glob("models/*")

unmaskerfile = open("models/unmasker.aimodel", "w+b")
summarizerfile = open("models/summarizer.aimodel", "w+b")
textgeneratorfile = open("models/textgenerator.aimodel", "w+b")
toxicfile = open("models/toxic.aimodel", "w+b")
classifierfile = open("models/classifier.aimodel", "w+b")
answerfile = open("models/answer.aimodel", "w+b")

if "models/unmasker.aimodel" in models:
    unmasker = pickle.load(unmaskerfile)
else:
    unmasker = transformers.pipeline("fill-mask", model="bert-base-uncased")
    pickle.dump(unmasker, unmaskerfile)

if "models/summarizer.aimodel" in models:
    summarizer = pickle.load(summarizerfile)
else:
    summarizer = transformers.pipeline("summarization", model="facebook/bart-large-cnn")
    pickle.dump(summarizer, summarizerfile)

if "models/textgenerator.aimodel" in models:
    textgenerator = pickle.load(textgeneratorfile)
else:
    textgenerator = transformers.pipeline("text-generation", model="gpt2")
    pickle.dump(textgenerator, textgeneratorfile)

if "models/toxic.aimodel" in models:
    toxic = pickle.load(toxicfile)
else:
    toxic = Detoxify("original")
    pickle.dump(toxic, toxicfile)

if "models/classifier.aimodel" in models:
    classifier = pickle.load(classifierfile)
else:
    classifier = transformers.pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    pickle.dump(classifier, classifierfile)

if "models/answer.aimodel" in models:
    answering = pickle.load(answerfile)
else:
    answering = transformers.pipeline("question-answering", model="deepset/xlm-roberta-base-squad2")
    pickle.dump(answering, answerfile)

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
        mask += "."
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
            description="The maximum length of the summarized text",
            required=False,
            default=130
        ),
        minLength: int = SlashOption(
            name="minimumlength",
            description="The minimum length of the summarized text",
            required=False,
            default=30
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


class Generate(ui.View):
    def __init__(self):
        super().__init__()
        self.value = None

    @ui.button(
        label="Generate more",
        style=ButtonStyle.green
    )
    async def generatemore(
            self,
            button: ui.Button,
            ctx: Interaction
    ):
        self.value = True
        self.stop()

    @ui.button(
        label="Variate",
        style=ButtonStyle.green
    )
    async def variate(
            self,
            button: ui.Button,
            ctx: Interaction
    ):
        self.value = False
        self.stop()


@bot.slash_command(
    name="generate",
    description="Generates text to add on to the text you input",
    dm_permission=True
)
async def generate(
        ctx: Interaction,
        text: str = SlashOption(
            name="text",
            description="The text to add on to"
        ),
        seed: int = SlashOption(
            name="seed",
            description="The seed used to generate the text",
            required=False,
            default=None
        ),
        maxLength: int = SlashOption(
            name="maxlength",
            description="The maximum length of the generated text",
            required=False,
            default=None
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    if maxLength is None:
        maxLength = text.split(" ")
        maxLength = len(maxLength) + 50
    if seed is None:
        usedSeed = random.randint(0, 99)
    else:
        usedSeed = seed
    transformers.set_seed(usedSeed)
    generated = textgenerator(text, max_length=maxLength, num_return_sequences=1)[0]["generated_text"]
    if len(generated) > 1811:
        await ctx.send("""```ansi
[2;31m[1;31mReturned text is more than 1811 characters. Did not execute.[0m[2;31m[0m
```""")
    else:
        generated = generated.replace(f"{text}", "")
        view = Generate()
        embed = Embed()
        embed.add_field(name="Seed:", value=usedSeed)
        msg = await ctx.send(
            f"""```ansi
{text}[2;40m[2;34m[2;41m[2;45m[2;30m[2;37m[2;47m[2;30m{generated}[0m[2;37m[2;47m[0m[2;37m[2;45m[0m[2;30m[2;45m[0m[2;34m[2;45m[0m[2;34m[2;41m[0m[2;34m[2;40m[0m[2;40m[0m
```""",
            embed=embed,
            view=view
        )
        await view.wait()
        if view.value is None:
            pass
        elif view.value:
            newMsg = await msg.reply("Generating")
            maxLength += 50
            newGenerated = textgenerator(text+generated, max_length=maxLength, num_return_sequences=1)[0]["generated_text"]
            newGenerated = newGenerated.replace(f"{text+generated}", "")
            await newMsg.edit(
                f"""```ansi
    {text+generated}[2;40m[2;34m[2;41m[2;45m[2;30m[2;37m[2;47m[2;30m{newGenerated}[0m[2;37m[2;47m[0m[2;37m[2;45m[0m[2;30m[2;45m[0m[2;34m[2;45m[0m[2;34m[2;41m[0m[2;34m[2;40m[0m[2;40m[0m
    ```"""
            )
        elif not view.value:
            newMsg = await msg.reply("Generating")
            seed = random.randint(0, 99)
            transformers.set_seed(seed)
            newGenerated = textgenerator(text, max_length=maxLength, num_return_sequences=1)[0]["generated_text"]
            newGenerated = newGenerated.replace(f"{text}", "")
            embed = Embed()
            embed.add_field(name="Seed:", value=seed)
            await newMsg.edit(
                f"""```ansi
{text}[2;40m[2;34m[2;41m[2;45m[2;30m[2;37m[2;47m[2;30m{newGenerated}[0m[2;37m[2;47m[0m[2;37m[2;45m[0m[2;30m[2;45m[0m[2;34m[2;45m[0m[2;34m[2;41m[0m[2;34m[2;40m[0m[2;40m[0m
```""",
                embed=embed
            )


@bot.slash_command(
    name="toxic",
    description="Tells you how toxic a message is",
    dm_permission=True
)
async def toxiccmd(
        ctx: Interaction,
        text: str = SlashOption(
            name="text",
            description="The text to classify"
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    result = toxic.predict(text)
    toxicity = str(round(result["toxicity"] * 100)) + "%"
    severe = str(round(result["severe_toxicity"] * 100)) + "%"
    obscene = str(round(result["obscene"] * 100)) + "%"
    threat = str(round(result["threat"] * 100)) + "%"
    insult = str(round(result["insult"] * 100)) + "%"
    identity = str(round(result["identity_attack"] * 100)) + "%"
    embed = Embed(title="Classification done:")
    embed.add_field(name="Original", value=f"{text}", inline=False)
    embed.add_field(name="Toxicity:", value=f"{toxicity}", inline=True)
    embed.add_field(name="Severe Toxicity:", value=f"{severe}", inline=True)
    embed.add_field(name="Obscene:", value=f"{obscene}", inline=True)
    embed.add_field(name="Threat:", value=f"{threat}", inline=True)
    embed.add_field(name="Insult:", value=f"{insult}", inline=True)
    embed.add_field(name="Identity Attack:", value=f"{identity}", inline=True)
    await ctx.send(embed=embed)


@bot.slash_command(
    name="classifier",
    description="Classifies text into different categories",
    dm_permission=True
)
async def classifiercmd(
        ctx: Interaction,
        text: str = SlashOption(
            name="text",
            description="The text to classify"
        ),
        tags: str = SlashOption(
            name="tags",
            description="The tags to classify the text with. SEPARATE WITH COMMAS"
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    tagsNoStrip = tags.split(',')
    tags = []
    for tag in tagsNoStrip:
        tags.append(tag.strip())
    result = classifier(text, tags, multi_class=True)
    embed = Embed(title="Classification done:")
    embed.add_field(name="Original:", value=f"{text}", inline=False)
    tagNum = 0
    for label in result["labels"]:
        percent = str(round(result["scores"][tagNum] * 100)) + "%"
        embed.add_field(name=f"{label}:", value=f"{percent}", inline=True)
        tagNum += 1
    await ctx.send(embed=embed)


@bot.slash_command(
    name="answer",
    description="Get an answer from a context",
    dm_permission=True
)
async def answer(
        ctx: Interaction,
        question: str = SlashOption(
            name="question",
            description="The question you want to answer"
        ),
        context: str = SlashOption(
            name="context",
            description="The context for the question"
        )
):
    interactionResponse = ctx.response
    await interactionResponse.defer(ephemeral=False, with_message=True)
    result = answering({"question": question, "context": context})
    score = str(round(result["score"] * 100)) + "%"
    correctanswer = result["answer"]
    embed = Embed(title="Answered:")
    embed.add_field(name="Question:", value=f"{question}", inline=False)
    embed.add_field(name="Context:", value=f"{context}", inline=False)
    embed.add_field(name="Answer:", value=f"{correctanswer}", inline=True)
    embed.add_field(name="Score:", value=f"{score}", inline=True)
    await ctx.send(embed=embed)

with open("token.txt", "r") as token:
    bot.run(token.read())
