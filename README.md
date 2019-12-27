# WaveFlow: Reimplementation of a compact flow-based model for raw audio

The original article can be found here https://arxiv.org/abs/1912.01219

This implementation contains both a fully parametric WaveFlow and a fast-generation algorithm adapted from https://arxiv.org/abs/1611.09482, and takes advantage of Nvidia APEX technology to produce audio at a very fast pace (sample generation frequency can go up to 400kHz, which is 25x real-time)
