# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(g_samples, g_targets, a_samples, a_targets, device):
    g_samples = g_samples.to(device, non_blocking=True)
     # print("G SAMPLES: ", g_samples)
    g_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in g_targets]
    # print("A SAMPLES: ", a_samples)
    a_samples = a_samples.to(device, non_blocking=True)
    a_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in a_targets]
    return g_samples, g_targets, a_samples, a_targets

class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.g_next_samples, self.g_next_targets, self.a_next_samples, self.a_next_targets = next(self.loader)
        except StopIteration:
            self.g_next_samples = None
            self.g_next_targets = None
            self.a_next_samples = None
            self.a_next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.g_next_samples, self.g_next_targets, self.a_next_samples, self.a_next_targets = to_cuda(self.g_next_samples, self.g_next_targets, self.a_next_samples, self.a_next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            g_samples = self.g_next_samples
            g_targets = self.g_next_targets
            a_samples = self.a_next_samples
            a_targets = self.a_next_targets
            if g_samples is not None:
                g_samples.record_stream(torch.cuda.current_stream())
            if g_targets is not None:
                for t in g_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            if a_samples is not None:
                a_samples.record_stream(torch.cuda.current_stream())
            if a_targets is not None:
                for t in a_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                g_samples, g_targets, a_samples, a_targets = next(self.loader)
                g_samples, g_targets, a_samples, a_targets = to_cuda(g_samples, g_targets, a_samples, a_targets, self.device)
            except StopIteration:
                g_samples = None
                g_targets = None
                a_samples = None
                a_targets = None
        return g_samples, g_targets, a_samples, a_targets
