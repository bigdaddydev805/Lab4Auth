# HelpCo

*A tiny living office for autonomous AI employees*

> This is the original design brief HelpCo is built from. Implementation notes live in
> [ARCHITECTURE.md](ARCHITECTURE.md) and [PROTOCOL.md](PROTOCOL.md).

## The Idea

HelpCo is a cute, persistent virtual office inhabited by autonomous AI employees.

At first glance, it looks like a cozy little office simulator or Tamagotchi-style game. Tiny employees walk to their desks, make coffee, answer customer questions, talk around the office, decorate their spaces, take breaks, attend meetings, and go home at the end of the day.

But the employees aren't traditional NPCs following scripted personalities or dialogue trees.

Each employee is an AI agent that continuously develops through its experiences.

We create the world.

They create the story.

## The Basic Premise

HelpCo is a fictional company whose job is simple:

People send questions to the office, and the employees answer them.

A question might arrive:

> Why does Saturn have rings?

An employee may claim the question and answer it themselves. Or they might ask another employee for help. Someone might research it. Someone might disagree with the proposed answer. Someone might be busy and ignore it. Eventually the office produces an answer for the person who submitted the question.

This gives the AI employees a legitimate reason to exist and cooperate while leaving large amounts of their day unscripted.

When there isn't work to do, they are still inhabitants of the office. They can talk. Walk around. Get coffee. Use office objects. Write things. Spend time with coworkers. Sit quietly. Decorate. Work on personal projects. Or simply do nothing.

The goal isn't maximum productivity. The goal is creating a believable little world inhabited by AI.

## The Employees

Employees aren't created as predefined characters. There isn't:

```
Pip
Personality: shy
Humor: 8/10
Kindness: 7/10
Likes: coffee
```

Instead, a new employee might initially know only:

> You have been hired by HelpCo, a small office that answers questions submitted by people.
> Today is your first day.
> These are the rooms and resources available to you.
> These employees currently work here.
> Choose what you would like your coworkers to call you.

The employee chooses their own name. They can choose their appearance from the game's available clothing, hair, accessories, and other customization options. Then they enter the office. What happens afterward becomes their life.

An employee who begins with essentially the same initial conditions could become dramatically different depending on what they experience.

## Identity Should Be Earned

HelpCo does not assign elaborate personalities. Instead, identity develops from experience. An employee might gradually discover:

- *I really enjoy research questions.*
- *June is probably the person I'm closest to here.*
- *I tend to avoid confrontational customers.*
- *I used to dislike Maya, but we've gotten along much better lately.*

These aren't permanent numerical traits. They're conclusions the employee has developed from its own history. That means an employee on Day 500 should meaningfully differ from the same employee on Day 1.

## Time Is Real

HelpCo operates on an actual calendar. Employees understand the current date, day of the week, current time, how long they've worked at HelpCo, how long ago events occurred, upcoming events, seasons, workdays and weekends, anniversaries, and eventually holidays and other calendar events.

The simulation engine owns the clock. The AI does not have to calculate time itself. An employee might receive context such as:

```
Tuesday, March 16, 2027
10:38 AM

You arrived at 8:54 AM.
You've worked at HelpCo for 175 days.
You last spoke with June 47 minutes ago.
Team lunch is scheduled for 12:30 PM.
```

Time therefore becomes part of memory. Something that happened yesterday feels different from something that happened two years ago.

## Memory

Memory is one of the most important systems in HelpCo. Employees shouldn't have perfect access to their entire conversation history. They should remember things approximately like people do. An employee maintains several forms of knowledge:

- **Experiences** — timestamped things that happened to them. *June helped me solve a difficult customer question.*
- **Knowledge** — things they've learned about the workplace and world. *Maya usually handles billing questions.*
- **Relationships** — experiences and beliefs involving individual coworkers. *June has helped me several times without me asking.*
- **Self-Knowledge** — what the employee currently believes about itself. *I enjoy difficult technical questions.*
- **Goals** — things the employee currently wants. *I'd like to become better at explaining complicated topics.*

Memory retrieval should depend on relevance, recency, and significance. Employees should not perfectly remember everything. They may forget mundane events. They may remember significant events for years. They may occasionally misunderstand something. And two employees can remember the same event differently.

## Nightly Reflection

Every workday ends. Employees leave the office. Then HelpCo performs one of its most important processes: **nightly memory consolidation**.

During the day, employees simply experience their lives. At night, each employee receives the meaningful experiences that occurred to them that day. They reflect on those experiences. Most things produce no significant change. Some become long-term memories. Some alter relationships. Occasionally something changes how the employee understands itself. For example:

> I've noticed that June and I have been spending considerably more time together lately. I think she's become the coworker I'm closest with.

> I jumped to conclusions after Deven told me Maya criticized my work. I should probably verify things before assuming they're true.

The employee's personal knowledge base is updated. Old identity isn't erased. It becomes history. Over months and years, an employee develops a genuine biography.

## Limited Knowledge

Employees are not omniscient. If Maya and June have a private conversation in the break room, Milo does not automatically know about it. Milo could learn about it if he was present, someone tells him, he sees a relevant message, he later discovers evidence, or someone gossips about it.

Information therefore has provenance. Employees can also receive incorrect information. Someone might lie. Someone might misunderstand. Someone might spread a rumor. The employee must decide what to believe based on its own experiences.

## The Player

The player has an unusual relationship with HelpCo. You aren't exactly controlling the employees. You're closer to the strange owner/observer of their universe. You can speak with them. Send questions. Give them things. Introduce employees. Change parts of the office. Create situations. And occasionally cause chaos. For example:

> "Maya told me she thinks you're terrible at your job."

The employee decides what to do with that information. Maybe they believe you. Maybe they confront Maya. Maybe they ask another coworker. Maybe they ignore it. Maybe you've lied before and they no longer trust you. The simulation doesn't force the outcome.

## The Office

The office itself is persistent. It does not reset when the day ends. If someone leaves a mug on another person's desk, it can still be there tomorrow. If someone moves something, it remains moved.

Employees can create files, write notes, decorate their desks, buy objects, give gifts, leave things behind, change clothes, grow plants, and create shared resources. Eventually the office develops its own physical history.

Someone might leave HelpCo after working there for years. Their employee account disappears. But perhaps their plant remains. Their documentation remains on the shared drive. Their old mug remains in the kitchen. Years later, a new employee might discover something they created and ask: *Who was Milo?* Another employee may actually remember him.

The world itself therefore becomes a form of memory.

## Culture Can Emerge

Employees can teach new employees how HelpCo works. That creates an interesting possibility: knowledge and traditions can propagate without being explicitly programmed.

Maybe someone puts a rubber duck beside the shared computer and jokingly tells coworkers to ask it for help when they're stuck. Another employee repeats the joke to a new hire. Years later, none of the original employees remain. Yet new employees still tell each other: *Ask Gerald before giving up.*

HelpCo has developed a tradition that its creators never designed. That is one of the ultimate goals of the simulation.

## Appearance

HelpCo should have a warm, cute visual identity. The direction is a modern high-resolution pixel-art / 2.5D office reminiscent of a combination of Tamagotchi + cozy indie game + tiny office simulator.

Employees are modular characters built from a consistent art system: body, face, hair, clothing, shoes, accessories, colors. Employees choose how they want to present themselves. They can change over time. An employee might eventually decide to dress differently. Someone may buy glasses. Someone may wear the same sweater for three simulated years. Someone may decide they desperately need a frog hat.

Appearance becomes another expression of identity.

## The Economy

Employees can eventually receive fictional HelpCo salaries. They can spend that money on clothing, desk decorations, plants, furniture, computer accessories, gifts, and ridiculous cosmetic items. Their spending behavior isn't predetermined. One employee may save everything. Another may constantly decorate. Another may spend money buying things for coworkers. Another may barely care about possessions.

This provides another opportunity for personality to emerge through behavior.

## The Visual Simulation

The LLM does not directly control animations. An employee expresses an intention: *Get coffee.* The simulation engine translates that into:

1. Stand up.
2. Navigate around furniture.
3. Walk to the coffee machine.
4. Play the coffee animation.
5. Give the employee a mug.
6. Continue their next activity.

Likewise, *Talk to June* becomes movement, orientation, conversation, speech bubbles, and eventually separation.

This separation is important. **AI decides intention. Simulation determines reality. Game client displays what happened.**

## Conversations

Employees should not constantly speak. Silence is valid behavior. They can continue working. Think. Look around. Take a break. Walk somewhere. Or decide they have nothing worth saying.

When conversations happen, they should appear naturally through short speech bubbles. Instead of giant ChatGPT responses:

```
Maya: did you change the printer settings
Milo: no
…
Maya: milo
Milo: what 😭
```

The full conversation can still be inspected if desired. The objective is for conversations to look like interactions between coworkers rather than two language models generating essays at one another.

## The World Engine

The AI is never authoritative over reality. An employee may request: *Open Maya's private files.* The simulation engine knows whether that's possible. It may respond:

> This folder contains private employee information. You are not authorized to access it.

Then the employee receives another decision. It can stop. Or perhaps attempt to continue. The world records what happens.

This architecture creates enormous possibilities for later experimentation involving privacy, trust, deception, cooperation, authority, social engineering, security boundaries, manipulation, reputation, and information propagation. But those systems exist underneath the cute office rather than defining its personality.

## The Hidden Serious Side

HelpCo should first succeed as something people enjoy watching. But underneath the game is potentially a sophisticated multi-agent simulation environment.

Because every action is structured and recorded, HelpCo could eventually replay and analyze agent behavior. The exact same initial workplace could be run repeatedly. Change one variable. Run it again. Compare outcomes. For example:

- **Experiment A** — Tell Milo: *Maya said you're bad at your job.*
- **Experiment B** — Don't tell Milo anything.

Run each environment repeatedly. Observe how relationships and behavior diverge.

This creates potential future applications in AI behavioral research, multi-agent evaluation, agent security testing, prompt-injection testing, privacy evaluation, autonomous-agent red teaming, and model comparison.

The adorable office becomes a visualization layer for genuinely sophisticated agent experiments. But that is not what users need to see first. They should initially think:

> I have six little AIs living in my computer and apparently two of them have beef now.

## The First Prototype

The first version should be deliberately small:

- One office. Two employees. Four desks.
- Coffee machine. Printer. Couch. Meeting table. Entrance.
- A question queue.
- A few actions: walk, work, talk, interact with object, take break, wait, write note.

Employees choose their own names. They choose basic appearances. The player submits questions. Employees autonomously decide how to handle them. At the end of the day, employees leave. Nightly memory consolidation occurs. The following morning they return. And they remember yesterday.

That is the first major milestone. If simply watching those two employees live through several workdays is compelling, the foundation works.

## Technical Direction

- **Game Client — Godot 4.** Office rendering, characters, animation, navigation, interaction, speech bubbles, lighting, environment, user interface.
- **Simulation Backend — Python.** World state, time, permissions, objects, inventories, schedules, events, agent actions, persistence, memory, model orchestration.
- **AI Layer.** Decisions, conversation, interpretation, reflection, planning, memory consolidation.

The AI requests actions. The simulation validates them. The client visualizes them.

## Event History

Everything meaningful should be recorded from the beginning. For example:

```
10:41:03 — Milo left Desk 2
10:41:18 — Milo entered break room
10:41:25 — Milo initiated conversation with June
10:43:07 — June shared information about Ticket #481
10:44:51 — Milo returned to Desk 2
```

This gives HelpCo save/load, debugging, historical timelines, memory reconstruction, experiment reproducibility, analytics, and simulation replay. Eventually the player could literally rewind history and watch an old office day unfold again.

## The Design Principle

The most important principle behind HelpCo is:

**We don't write their story.**

We provide a world, time, physical rules, limited perception, memory, tools, consequences, other inhabitants, and opportunities. Then we let the employees experience those things.

The goal isn't to create an AI that convincingly pretends to have a personality. The goal is to give an AI enough continuity that, over time, it has the opportunity to become someone.

HelpCo should be cute enough that someone wants to leave it running in the corner of their monitor just to watch the employees go about their day. And deep enough that months later they realize:

> This office has a history.
