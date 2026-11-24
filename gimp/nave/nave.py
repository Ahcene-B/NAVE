#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
GIMP 3 plugin: NAVE Superpixels  (Filters > Generic > NAVE Superpixels...)

Install as  <GIMP profile>/plug-ins/nave/  containing nave.py, nave_env.py,
nave_core.py, nave_models.py and graphs/. Only the standard library is used
here; the work happens in nave_core.py, run in a private Python environment.
"""
import json
import os
import subprocess
import sys
import tempfile

import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, Gegl, GObject, GLib, Gtk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nave_env  # noqa: E402
import nave_mem  # noqa: E402

PROC = "plug-in-nave-superpixels"
RW = GObject.ParamFlags.READWRITE
RESNET_BLOCKS = range(1, 5)
VIT_BLOCKS = range(1, 13)
OUTPUTS = ["out_borders", "out_colors", "out_segmentation"]
LAYOUT = ["model", "training", "resnet-blocks", "vit-blocks", "window_size", "two_pass", "upsample",
          "clusters", "outputs", "border_width", "seed", "python"]


def _choice(items):
    c = Gimp.Choice.new()
    for i, (nick, label) in enumerate(items):
        c.add(nick, i, label, "")
    return c


def _lin_to_srgb(v):
    v = min(max(v, 0.0), 1.0)
    v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return int(round(v * 255))


def _status(text):
    Gimp.progress_set_text(text)


def _pulse():
    Gimp.progress_pulse()


def _ask_yes_no(message):
    dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.QUESTION, buttons=Gtk.ButtonsType.YES_NO,
                               text="Set up Python for NAVE?")
    dialog.format_secondary_text(message)
    answer = dialog.run()
    dialog.destroy()
    return answer == Gtk.ResponseType.YES


def _ask_continue(message):
    dialog = Gtk.MessageDialog(message_type=Gtk.MessageType.WARNING, buttons=Gtk.ButtonsType.OK_CANCEL,
                               text="NAVE may run out of memory")
    dialog.format_secondary_text(message + "\n\nRun anyway?")
    answer = dialog.run()
    dialog.destroy()
    return answer == Gtk.ResponseType.OK


def _run_worker(python, rgb, w, h, params):
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.raw")
        job, errp = os.path.join(td, "job.json"), os.path.join(td, "err.txt")
        outputs = {n: os.path.join(td, f"{n}.raw") for n in ("colors", "segmentation", "borders")}
        with open(inp, "wb") as f:
            f.write(rgb)
        with open(job, "w") as f:
            json.dump(dict(input=inp, outputs=outputs, width=w, height=h, params=params), f)
        with open(errp, "w") as err:
            proc = subprocess.Popen([python, os.path.join(HERE, "nave_core.py"), job],
                                    stdout=subprocess.PIPE, stderr=err, text=True,
                                    creationflags=nave_env.NO_WINDOW)
            for line in proc.stdout:
                if line.startswith("PROGRESS"):
                    parts = line.rstrip("\n").split(" ", 2)
                    Gimp.progress_update(float(parts[1]))
                    if len(parts) > 2 and parts[2]:
                        Gimp.progress_set_text(parts[2])
            proc.wait()
        if proc.returncode != 0:
            with open(errp) as f:
                lines = f.read().strip().splitlines()
            if proc.returncode < 0 or proc.returncode in (137, 3221225495) or not lines:
                raise RuntimeError("the NAVE worker was stopped, most likely because the computer ran out of "
                                   "memory. Try a smaller window size or fewer blocks.")
            raise RuntimeError(lines[-1])
        layers = {}
        for name, path in outputs.items():
            if os.path.exists(path):
                with open(path, "rb") as f:
                    layers[name] = f.read()
        return layers


class NavePlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return [PROC]

    def do_set_i18n(self, name):
        return False

    def do_create_procedure(self, name):
        p = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
        p.set_image_types("RGB*")
        p.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        p.set_menu_label("NAVE Superpixels...")
        p.add_menu_path("<Image>/Filters/Generic")
        p.set_documentation("Superpixels from a vision encoder's features (NAVE)",
                            "Clusters multi-depth encoder activations in overlapping 448 px windows and "
                            "adds layers with segment borders, colored superpixels and/or the NAVE "
                            "cluster segmentation.", name)
        p.set_attribution("NAVE plugin", "NAVE plugin", "2026")

        p.add_choice_argument("model", "_Model", "Backbone",
                              _choice([("resnet18", "ResNet18"), ("resnet50", "ResNet50"), ("vits16", "ViT-S/16")]),
                              "resnet50", RW)
        self.training_choice = _choice([("random", "Random"), ("imagenet", "ImageNet"), ("dino", "DINO")])
        p.add_choice_argument("training", "_Training",
                              "Weights (pretrained ones are downloaded on first use; no DINO for ResNet18)",
                              self.training_choice, "imagenet", RW)
        for i in RESNET_BLOCKS:
            p.add_boolean_argument(f"resnet_block{i}", f"ResNet {i}",
                                   f"ResNet: use the output of block {i} (layer{i})", i >= 3, RW)
        for i in VIT_BLOCKS:
            p.add_boolean_argument(f"vit_block{i}", f"ViT {i}",
                                   f"ViT: use the output of transformer block {i}", i == 12, RW)
        p.add_choice_argument("upsample", "_Upsampling", "Interpolation of features and segment boundaries",
                              _choice([("nearest", "Nearest"), ("linear", "Linear"), ("cubic", "Cubic")]),
                              "cubic", RW)
        p.add_int_argument("window_size", "_Window size",
                           "Size of the overlapping windows in px (multiple of 32; larger uses more memory)",
                           224, 2048, 448, RW)
        p.add_boolean_argument("two_pass", "_Two passes",
                               "Cluster the whole image at once (runs the network twice); "
                               "needed for the segmentation layer", False, RW)
        p.add_int_argument("clusters", "_Clusters (K)",
                           "k-means clusters per window, or for the whole image with two passes", 2, 64, 10, RW)
        p.add_boolean_argument("out_borders", "_Borders", "Layer with segment borders (foreground color)", True, RW)
        p.add_boolean_argument("out_colors", "_Colored superpixels",
                               "Layer with each segment filled with its average color", False, RW)
        p.add_boolean_argument("out_segmentation", "_Segmentation",
                               "Layer with each pixel colored by its NAVE cluster (two passes only)", False, RW)
        p.add_int_argument("border_width", "Border _width", "Border line width in pixels", 1, 50, 1, RW)
        p.add_int_argument("seed", "_Seed", "Seed for k-means and random weights", 0, 1000000, 0, RW)
        p.add_string_argument("python", "External _Python",
                              "Optional: a Python that has numpy and onnxruntime. Empty = set up automatically.",
                              "", RW)
        return p

    def build_dialog(self, dialog, procedure, config):
        """Block checkboxes in two grids; only the grid for the chosen model is shown.
        Falls back to showing every option if this GIMP lacks a needed call."""
        def flowbox(box_id, ids):
            try:
                return dialog.fill_flowbox(box_id, ids)
            except (AttributeError, TypeError):
                return dialog.fill_flowbox_from_list(box_id, ids)
        try:
            grids = {"resnet": flowbox("resnet-blocks", [f"resnet_block{i}" for i in RESNET_BLOCKS]),
                     "vit": flowbox("vit-blocks", [f"vit_block{i}" for i in VIT_BLOCKS])}
            flowbox("outputs", OUTPUTS)
            dialog.fill(LAYOUT)
        except Exception:
            dialog.fill(None)                           # static layout: all options visible
            grids = {}
        for prop, source in (("out_segmentation", "two_pass"), ("border_width", "out_borders")):
            try:                                        # greyed out unless `source` is ticked
                dialog.set_sensitive(prop, True, config, source, False)
            except Exception:
                pass

        def no_segmentation(*_):
            if not config.get_property("two_pass") and config.get_property("out_segmentation"):
                config.set_property("out_segmentation", False)
        config.connect("notify::two-pass", no_segmentation)
        no_segmentation()

        choice = getattr(self, "training_choice", None)
        if choice is None:
            try:
                choice = Gimp.param_spec_choice_get_choice(procedure.find_argument("training"))
            except Exception:
                choice = None

        def update(*_):
            model = config.get_property("model")
            family = "resnet" if model.startswith("resnet") else "vit"
            for name, grid in grids.items():
                try:
                    if name == family:
                        grid.set_no_show_all(False)
                        grid.show_all()
                    else:
                        grid.set_no_show_all(True)       # stays hidden when the dialog is shown
                        grid.hide()
                except Exception:
                    pass
            if model == "resnet18" and config.get_property("training") == "dino":
                config.set_property("training", "imagenet")
            if choice is not None:
                try:
                    choice.set_sensitive("dino", model != "resnet18")
                except Exception:
                    pass

        config.connect("notify::model", update)
        update()

    def run(self, procedure, run_mode, image, drawables, config, data):
        def fail(msg):
            Gimp.message(f"NAVE: {msg}")
            return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(msg))

        if len(drawables) != 1:
            return fail("select exactly one layer.")
        interactive = run_mode == Gimp.RunMode.INTERACTIVE
        if interactive:
            GimpUi.init(PROC)
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "NAVE Superpixels")
            self.build_dialog(dialog, procedure, config)
            ok = dialog.run()
            dialog.destroy()
            if not ok:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())

        get = config.get_property
        model = get("model")
        params = {
            "model": model,
            "training": get("training"),
            "blocks": "",
            "upsample": get("upsample"),
            "clusters": get("clusters"),
            "seed": get("seed"),
            "border_width": get("border_width"),
            "window": min(2048, max(224, int(round(get("window_size") / 32)) * 32)),
            "two_pass": bool(get("two_pass")),
            "outputs": [o[4:] for o in OUTPUTS if get(o)],
        }
        if not params["outputs"]:
            return fail("tick at least one output layer.")
        if "segmentation" in params["outputs"] and not params["two_pass"]:
            return fail("the segmentation layer needs Two passes.")
        if model.startswith("resnet"):
            chosen = [str(i) for i in RESNET_BLOCKS if get(f"resnet_block{i}")]
            params["blocks"] = "N" + "".join(chosen)
        else:
            chosen = [str(i) for i in VIT_BLOCKS if get(f"vit_block{i}")]
            params["blocks"] = "N:" + "-".join(chosen)
        if not chosen:
            return fail("tick at least one block.")
        if model == "resnet18" and params["training"] == "dino":
            return fail("there are no DINO weights for ResNet18; choose ImageNet or Random.")
        r, g, b, _ = Gimp.context_get_foreground().get_rgba()     # linear light
        params["border_color"] = [_lin_to_srgb(r), _lin_to_srgb(g), _lin_to_srgb(b), 255]

        drawable = drawables[0]
        w, h = drawable.get_width(), drawable.get_height()
        blocks = [int(c) for c in params["blocks"].lstrip("N:").replace("-", "")] if model.startswith("resnet") \
            else [int(c) for c in params["blocks"][2:].split("-")]
        warn = nave_mem.warning(model, blocks, params["window"], w, h, params["clusters"],
                                params["two_pass"], len(params["outputs"]))
        if warn:
            if interactive and not _ask_continue(warn):
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
            if not interactive:
                Gimp.message("NAVE: " + warn)

        Gimp.progress_init("NAVE superpixels")
        try:
            python = nave_env.ensure_python((get("python") or "").strip(),
                                            ask_uv=_ask_yes_no if interactive else None,
                                            status=_status, pulse=_pulse)
        except nave_env.SetupError as e:
            return fail(str(e))

        rect = Gegl.Rectangle.new(0, 0, w, h)
        rgb = drawable.get_buffer().get(rect, 1.0, "R'G'B' u8", Gegl.AbyssPolicy.CLAMP)
        try:
            layers = _run_worker(python, bytes(rgb), w, h, params)
        except Exception as e:
            return fail(str(e))

        image.undo_group_start()
        settings = f"{model} {params['training']} {params['blocks']} K{params['clusters']}"
        offsets = drawable.get_offsets()
        below = drawable                                # each new layer goes above the previous one
        names = {"colors": "colored superpixels", "segmentation": "segmentation", "borders": "borders"}
        for kind in ("colors", "segmentation", "borders"):   # borders end up on top
            if kind not in layers:
                continue
            layer = Gimp.Layer.new(image, f"NAVE {names[kind]} {settings}", w, h, Gimp.ImageType.RGBA_IMAGE,
                                   100.0, Gimp.LayerMode.NORMAL)
            image.insert_layer(layer, drawable.get_parent(), image.get_item_position(below))
            below = layer
            layer.set_offsets(offsets[-2], offsets[-1])
            buf = layer.get_buffer()
            buf.set(rect, "R'G'B'A u8", layers[kind])
            buf.flush()
            layer.update(0, 0, w, h)
        image.undo_group_end()
        Gimp.displays_flush()
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(NavePlugin.__gtype__, sys.argv)
