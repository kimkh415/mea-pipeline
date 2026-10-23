# What the metrics mean

Plain-language notes for the columns of `compiled.wide.csv`. Exact
definitions and settings: [README → Metrics](README.md#metrics).

Each value is for one condition of one recording, measured over the last
450 s of that condition (the whole condition if shorter; `t_start_s` and
`t_end_s` give the span). Values for single units are averaged over all kept
units; values for pairs are averaged over all pairs.

## How much units fire

- **`firingRate`, `firingRateISI`**: how fast a unit fires *while it is
  firing*: 1 / the average gap between its spikes. A unit that fires 2 spikes
  10 ms apart and then nothing scores 100 Hz, so a few such units can pull
  the average up.
- **`firingRateCount`**: spikes divided by the window length. Counts silent
  stretches too, so it is lower than `firingRateISI` for units that pause.
  Compare the two: if they disagree a lot, a few sparse units are driving
  `firingRateISI`.

## How regular the firing is

- **`fanoFactor`**: chop the window into 100 ms bins, count spikes per bin,
  and divide the variance of those counts by their mean. Random (Poisson)
  firing gives 1. Below 1, the counts are more even than random: the unit
  fires like a clock. Above 1, the counts swing more than random: spikes come
  in clumps (bursts), with quiet bins between.
  Caveat: with a fixed 100 ms bin the value also depends on firing rate. At
  ~1 Hz a bin is a tenth of the usual gap between spikes, so almost every bin
  holds 0 or 1 spike, and that gives ≈ 1 for any unit, regular or random.
  Values near 1 therefore do not prove random firing, and a drug that changes
  the rate changes this value too. Rajdl, Lansky & Kostal, *Front. Comput.
  Neurosci.* 14, 569049 (2020) remove the rate dependence by sizing the bin in
  multiples of each unit's mean gap ("operational time"): that is
  `fanoFactorOp`.
- **`fanoFactorOp`**: the same idea, but each unit's bin is 5 of its own
  average gaps between spikes, so a fast and a slow unit are judged on the
  same footing, and a drug that only speeds firing up leaves it unchanged.
  Same reading: ≈ 1 random, < 1 clock-like, > 1 clumped. A unit counts only
  if its bin fits at least 10 times in the span, i.e. its average gap is at
  most 1/50 of the span (9 s in 450 s). Use this one to compare conditions.
  Rajdl, Lansky & Kostal (2020).
- **`fanoOpUnits`**: how many units passed that rule; the others fire too
  slowly to measure this way.
- **`lvr`**: compares each gap between spikes with the next one. Steady gaps
  give < 1, random gaps ≈ 1, a mix of very short and very long gaps (bursting)
  > 1. Less affected by slow changes in rate than `fanoFactor`. Shinomoto,
  Shima & Tanji, *Neural Comput.* 15, 2823 (2003).
- **`fanoFactorCV2`**: the spread of the gaps between spikes (SD / mean,
  squared). Common in MEA papers, but it shifts with firing rate, so read it
  with care when a drug changes the rate.
- **`entropy`**: how varied a unit's spike counts per 30 ms bin are. Low: the
  unit mostly does the same thing (e.g. silent). High: many different counts.

## How much units fire together

- **`sttc`** (spike-time tiling coefficient): for two units, how often a spike
  of one has a spike of the other within ±100 ms, corrected for how much each
  unit fires (busy units would overlap by chance). 0 = no more overlap than
  chance; 1 = every spike of each has a partner; negative = they overlap less
  than chance, as if avoiding each other. Cutts & Eglen, *J. Neurosci.* 34,
  14288 (2014).
- **`sttc_dt15`, `sttc_dt100`**: the same at ±15 ms (tight, near-simultaneous
  firing) and ±100 ms (loose co-activation).
- **`sttcNegativeFraction`**: share of pairs that overlap less than chance.
- **`mutualInfo`**: for two units, count spikes in the same 30 ms bins and
  ask how much one unit's count tells you about the other's, in bits
  (mutual information of the two count sequences). 0 = independent.

## Bursts of single units

A burst is a run of at least 3 spikes with short gaps (it starts below
200 ms and ends at a gap of 300–500 ms): the Maximum Interval method, as in
meaRtools (Gelfman et al., *PLoS Comput. Biol.* 14, e1006506, 2018). A unit
with at least 5 bursts is *bursting*; the others are *tonic*.

- **`n_bursting`, `n_tonic`, `burst.neurons`, `tonic.neurons`**: how many
  units are bursting / tonic, as counts and fractions.
- **`burst.rate_mean`**: bursts per minute.
- **`burst.duration_mean`**: how long a burst lasts (s).
- **`burst.amplitude_mean`**: how fast the unit fires inside a burst (Hz).
- **`burst.nSpikes_norm`**: share of a unit's spikes that fall inside bursts.
- **`burst.rate_ff`**: how irregular the gaps between bursts are.
- **`nonburst.*`**: firing of bursting units between their bursts.
- **`tonic.*`**: firing of the tonic units.
- **`_cv`** on any of these: how much it varies from burst to burst (or unit
  to unit), as SD / mean.

## Network bursts

A network burst (event) is a moment when at least 20 % of all units fire
within the same 100 ms. Events less than 1 s apart count as one.

- **`network.nBurst`**: number of events. Blank = the condition is not network
  bursting; fewer than 3 = the other network values rest on very few events.
- **`network.n_channels`**: units that take part in events more than chance.
- **`network.background_fraction`**: share of units active between events.
- **`network.background_flag`**: 1 = so much background firing that chance
  overlap could look like an event; treat this condition's events with care.

## Families of network bursts

Network bursts are sorted into groups ("families") of events that look
alike: similar length, height and set of units. Each family gets its own
columns, `family1.`, `family2.`, ….

- **`families.n_cluster`**: how many families.
- **`families.quality`**: 2 = the split into families is clearly real, 1 = a
  smaller split is, 0 = there is just one kind of event.
- **`familyN.nBurst`**: events in this family; `.nBurstTotal`: in the
  condition.
- **`familyN.rate`**: events per minute.
- **`familyN.duration`**: how long an event lasts (s).
- **`familyN.amplitude`**: how fast the participating units fire during an
  event (Hz).
- **`familyN.height`**: how sharp the event's peak is.
- **`familyN.nSpikes.mean`**: spikes each participating unit fires per event.
- **`familyN.neuronsAct`**: share of participating units that join a given
  event.
- **`familyN.delay`**: how long after the event starts each unit fires its
  first spike (ms): who leads, who follows.
- **`familyN.distance`**: how spread out in time the units' spikes are
  around the peak (ms): tight or smeared events.
- **`familyN.nNeuron.*`**: how many units take part, in all events and in this
  family.
