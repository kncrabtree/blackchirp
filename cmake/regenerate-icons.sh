#!/bin/sh
# regenerate-icons.sh — rebuild platform icon artifacts from the SVG source.
#
# Inputs:
#   icons/blackchirp_icon.svg    (editable source)
#
# Outputs:
#   icons/blackchirp.ico                     (Windows)
#   icons/blackchirp.icns                    (macOS bundle)
#   icons/blackchirp.png                     (1024×1024, legacy linux pixmap)
#   icons/hicolor/<size>/apps/blackchirp.png (linux XDG icon theme)
#   icons/hicolor/scalable/apps/blackchirp.svg
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

# ---- Windows .ico ---------------------------------------------------------
ICO_SIZES="16 24 32 48 64 128 256"
ICO_INPUTS=""
for sz in ${ICO_SIZES}; do
    ICO_INPUTS="${ICO_INPUTS} ${TMPDIR}/icon_${sz}.png"
done
echo "Building blackchirp.ico..."
# shellcheck disable=SC2086
${IM} ${ICO_INPUTS} "${ICON_DIR}/blackchirp.ico"

# ---- macOS .icns ----------------------------------------------------------
# png2icns picks the format slot for each input by its dimensions.
ICNS_SIZES="16 32 48 128 256 512 1024"
ICNS_INPUTS=""
for sz in ${ICNS_SIZES}; do
    ICNS_INPUTS="${ICNS_INPUTS} ${TMPDIR}/icon_${sz}.png"
done
echo "Building blackchirp.icns..."
# shellcheck disable=SC2086
png2icns "${ICON_DIR}/blackchirp.icns" ${ICNS_INPUTS} >/dev/null

# ---- Linux hicolor tree ---------------------------------------------------
HICOLOR_SIZES="16 22 24 32 48 64 128 256 512"
echo "Building hicolor tree..."
for sz in ${HICOLOR_SIZES}; do
    dst="${HICOLOR_DIR}/${sz}x${sz}/apps"
    mkdir -p "${dst}"
    cp "${TMPDIR}/icon_${sz}.png" "${dst}/blackchirp.png"
done
mkdir -p "${HICOLOR_DIR}/scalable/apps"
cp "${SRC_SVG}" "${HICOLOR_DIR}/scalable/apps/blackchirp.svg"

# ---- Legacy /usr/share/pixmaps fallback ----------------------------------
cp "${TMPDIR}/icon_1024.png" "${ICON_DIR}/blackchirp.png"

echo "Done."
