#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os, sys
import pickle
import json
import string

from pynput.keyboard import Key, Listener

from ppaquette_gym_super_mario import wrappers

import gym
import gym_pull
from gym.envs.classic_control import rendering


action = 0
ckeys = set()
str2key = {
  'alt': Key.alt,
  'alt_gr': Key.alt_gr,
  'alt_l': Key.alt_l,
  'alt_r': Key.alt_r,
  'backspace': Key.backspace,
  'caps_lock': Key.caps_lock,
  'cmd': Key.cmd,
  'cmd_l': Key.cmd_l,
  'cmd_r': Key.cmd_r,
  'ctrl': Key.ctrl,
  'ctrl_l': Key.ctrl_l,
  'ctrl_r': Key.ctrl_r,
  'delete': Key.delete,
  'down': Key.down,
  'end': Key.end,
  'enter': Key.enter,
  'esc': Key.esc,
  'f1': Key.f1,
  'home': Key.home,
  'left': Key.left,
  'page_down': Key.page_down,
  'page_up': Key.page_up,
  'right': Key.right,
  'shift': Key.shift,
  'shift_l': Key.shift_l,
  'shift_r': Key.shift_r,
  'space': Key.space,
  'tab': Key.tab,
  'up': Key.up,
}

def read_json_keymap(fn):
  with open(fn) as df:
    return json.load(df)

def to_keymap(obj):
  keymap = dict()
  for key, val in enumerate(obj):
    keys = list()
    key = key.translate(str.maketrans('', '', string.whitespace))
    keystrs = key.split('+')
    for ks in keystrs:
      k = str2key.get(ks, '0')
      keys.append(k)
    keymap.set(frozenset(keys), int(val))
  return keymap

def on_press(key):
  global ckeys
  ckeys.add(key)

def on_release(key):
  global ckeys
  ckeys.remove(key)
  if key == Key.esc:
    return False

def trackdemo(gym_id, path='.'):
  global ckeys
  env = gym.make(gym_id)
  mode_wrapper = wrappers.SetPlayingMode('algo')
  ac_wrapper = wrappers.ToDiscrete()
  env = ac_wrapper(env)
  env = mode_wrapper(env)
  env.render()

  try:
    action = 0
    with Listener(on_press=on_press, on_release=on_release) as listener:
      demo_id = 0
      terminal = True
      demonstrations = list()
      while True:
        action = keymap.get(frozenset(ckeys), 0)
        if terminal:
          state = env.reset()
          with open(os.path.join(path, 'demo_{}.p'.format(demo_id)), 'wb') as f:
            pickle.dump(demonstrations, f, protocol=pickle.HIGHEST_PROTOCOL)
          demo_id += 1
          demonstrations = list()
        state, reward, terminal, _ = env.step(action)
        demonstrations.append(dict(state=state,
                                   action=action,
                                   reward=reward,
                                   terminal=terminal,
                                   internal=[]))
  except (KeyboardInterrupt, EOFError):
    listener.join()
    pass

  demo_id += 1
  with open(os.path.join(path, 'demo_{}.p'.format(demo_id)), 'wb') as f:
    pickle.dump(demonstrations, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(path):
  demos = list()
  for fn in os.listdir(path):
    with open(os.path.join(path, fn), 'rb') as f:
      demos += pickle.load(f)
  return demos

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('gym_id', help="ID of the gym environment, i.e. ppaquette/SuperMarioBros-1-1-v0")
  parser.add_argument('-s', '--save', type=str, help="Save demos to this directory")
  parser.add_argument('-k', '--keymap', type=str, help="Keymap json file")
  args = parser.parse_args()

  keymap = to_keymap(read_json_keymap(args.keymap))
  trackdemo(args.gym_id, path=args.save)

if __name__ == '__main__':
  main()
