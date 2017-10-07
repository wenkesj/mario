teaching mario to play mario
============================

This is a very simple example using [reinforceio/tensorforce](https://github.com/reinforceio/tensorforce),
`tensorflow` and `openai/gym`. This was a small weekend project when DQfD was first published and I thought
others might use it as a positive-slope for their experiments getting started. I don't have the access to
computing power to let this little guy run for long periods of time. Most of the agents I've trained
here make use of my own demonstrations playing `mario` and they are sparse. I must say, I think the agent is
just about as good as I am (that's definitely not saying much). I eventually want to combine this
technique with evolution strategies, since both frameworks are very straight-forward, scalable,
and easy to implement. It would be interesting to combine the weighted ranking algorithm of evolution strategies
with replay prioritization to make the next generation of parameters a function of the demonstrations
of past generations (if that makes sense, more on that later).

<div align="center">
  <img src="/assets/conv.gif"/>
  <p align="center">
    <strong>cnn</strong>
  </p>
</div>

<div align="center">

  <img src="/assets/lstm.gif"/>
  <p align="center">
    <strong>cnn+lstm</strong>
  </p>
</div>

If you want specific access to more documentation and details, please send me an email and I may be able to help!

**NOTE:** that this will probably not run on your system because it required a few changes to `openai/gym`
and `ppaquette/gym-super-mario`, as well as `reinforceio/tensorforce` :(. I basically steem-rolled
the errors that came up (`openai/gym` changes were due to a depreciation of an Env Wrapper that
`ppaquette/gym-super-mario` used. I chose to remedy this issue by cherrypicking the changes and placing
it into `ppaquette/gym-super-mario`, since it was removed without depreciation warning.)

The `reinforceio/tensorforce` changes are a little more complicated, since they involved a small DQFDAgent
change and a few code fixes - I will most likely submit a pull request for most of these libraries
so everyone can use this.

attempt to install
==================

```sh
# install/upgrade tensorforce
pip install --upgrade tensorforce
# install gym extensions
pip install gym_pull
# install mario environment
pip install ppaquette_gym_super_mario
python -c "import gym_pull; gym_pull.pull('github.com/ppaquette/gym-super-mario')"
```

attempt to teach/train and validate
===================================

```sh
# if your starting from scratch
mkdir agents demos monitors
# now run this, change if necessary
python mario.py ppaquette/meta-SuperMarioBros-v0 \
  -a DQFDAgent -c mario_agent.json -pt \
  -ld ./demos/ -s ./agents/ -m ./monitors/ -mv 100 -D
```

recording how bad you are at playing the game
=============================================

```sh
python demo.py ppaquette/meta-SuperMarioBros-v0 -s .
# start playing... it loops over when you die and starts a new demo file.
```
