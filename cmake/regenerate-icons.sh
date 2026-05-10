#!/bin/sh
# regenerate-icons.sh — rebuild platform icon artifacts from the SVG source.
#
# Inputs:
#   icons/blackchirp_icon.svg    (editable source)
#
# Outputs (acquisition app):
#   icons/blackchirp.ico                     (Windows)
#   icons/blackchirp.icns                    (macOS bundle)
#   icons/blackchirp.png                     (1024×1024, legacy linux pixmap)
#   icons/hicolor/<size>/apps/blackchirp.png (linux XDG icon theme)
#   icons/hicolor/scalable/apps/blackchirp.svg
#
# Outputs (viewer): the same set of artifacts under the
# blackchirp-viewer base name, with the rendered PNGs run through an
# RGB-channel negate (alpha preserved) so the viewer's taskbar/dock
# icon is visually distinct from the acquisition app's. The scalable
# SVG is intentionally not duplicated for the viewer — every desktop
# we target has hicolor PNGs up to 512 already.
#
# Required tools: inkscape, magick (or convert), png2icns.
# Run from the project root: cmake/regenerate-icons.sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
SRC_SVG="${ROOT}/icons/blackchirp_icon.svg"
ICON_DIR="${ROOT}/icons"
HICOLOR_DIR="${ICON_DIR}/hicolor"

if [ ! -f "${SRC_SVG}" ]; then
    echo "error: source SVG not found: ${SRC_SVG}" >&2
    exit 1
fi

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required tool not found: $1" >&2
        exit 1
    fi
}

need inkscape
need png2icns

if command -v magick >/dev/null 2>&1; then
    IM=magick
elif command -v convert >/dev/null 2>&1; then
    IM=convert
else
    echo "error: ImageMagick (magick or convert) not found" >&2
    exit 1
fi

# Sizes used across platforms. The union is rendered once into a temp dir, then
# the per-platform packers pick the sizes they want.
SIZES="16 22 24 32 48 64 128 256 512 1024"

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

echo "Rendering PNGs from ${SRC_SVG##*/}..."
for sz in ${SIZES}; do
    inkscape --export-type=png \
             --export-filename="${TMPDIR}/icon_${sz}.png" \
             --export-width="${sz}" \
             --export-height="${sz}" \
             "${SRC_SVG}" >/dev/null 2>&1
    printf '  %4dx%-4d\n' "${sz}" "${sz}"
done

# Inverted-color companions for the viewer. `-channel RGB -negate`
# operates only on the color channels so the alpha channel's pixel
# values are not negated; the `PNG32:` output prefix forces a 4-channel
# RGBA encoding so ImageMagick doesn't silently downgrade the result to
# Grayscale (dropping alpha) when it detects the rendered logo happens
# to be monochrome.
echo "Inverting PNGs for blackchirp-viewer..."
for sz in ${SIZES}; do
    ${IM} "${TMPDIR}/icon_${sz}.png" \
          -channel RGB -negate \
          "PNG32:${TMPDIR}/icon_${sz}_inv.png"
done

# Render acquisition-app and viewer artifacts in matched pairs so the
# inputs/outputs stay in lockstep when the size lists change.
#
# Args: <suffix> <output-base>
#   suffix      empty for the acquisition app, "_inv" for the viewer
#   output-base "blackchirp" or "blackchirp-viewer"
build_variant() {
    suffix=$1
    base=$2

    # ---- Windows .ico -----------------------------------------------------
    ICO_SIZES="16 24 32 48 64 128 256"
    ICO_INPUTS=""
    for sz in ${ICO_SIZES}; do
        ICO_INPUTS="${ICO_INPUTS} ${TMPDIR}/icon_${sz}${suffix}.png"
    done
    echo "Building ${base}.ico..."
    # shellcheck disable=SC2086
    ${IM} ${ICO_INPUTS} "${ICON_DIR}/${base}.ico"

    # ---- macOS .icns ------------------------------------------------------
    # png2icns picks the format slot for each input by its dimensions.
    ICNS_SIZES="16 32 48 128 256 512 1024"
    ICNS_INPUTS=""
    for sz in ${ICNS_SIZES}; do
        ICNS_INPUTS="${ICNS_INPUTS} ${TMPDIR}/icon_${sz}${suffix}.png"
    done
    echo "Building ${base}.icns..."
    # shellcheck disable=SC2086
    png2icns "${ICON_DIR}/${base}.icns" ${ICNS_INPUTS} >/dev/null

    # ---- Linux hicolor tree -----------------------------------------------
    HICOLOR_SIZES="16 22 24 32 48 64 128 256 512"
    echo "Building hicolor tree (${base})..."
    for sz in ${HICOLOR_SIZES}; do
        dst="${HICOLOR_DIR}/${sz}x${sz}/apps"
        mkdir -p "${dst}"
        cp "${TMPDIR}/icon_${sz}${suffix}.png" "${dst}/${base}.png"
    done

    # ---- Legacy /usr/share/pixmaps fallback ------------------------------
    cp "${TMPDIR}/icon_1024${suffix}.png" "${ICON_DIR}/${base}.png"
}

build_variant "" "blackchirp"
build_variant "_inv" "blackchirp-viewer"

# Scalable SVG only for the acquisition app — the viewer's PNGs cover
# every size the icon-theme spec asks for. Inverting an SVG cleanly
# (preserving stroke colors, gradients, and painted-over text) is more
# nuanced than `-channel RGB -negate` and isn't worth the maintenance
# cost when no consumer needs it.
mkdir -p "${HICOLOR_DIR}/scalable/apps"
cp "${SRC_SVG}" "${HICOLOR_DIR}/scalable/apps/blackchirp.svg"

echo "Done."
