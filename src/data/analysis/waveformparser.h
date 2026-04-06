#ifndef WAVEFORMPARSER_H
#define WAVEFORMPARSER_H

#include <vector>

#include <QtGlobal>

#include <data/experiment/digitizerconfig.h>
#include <data/storage/waveformbuffer.h>

namespace BC::Analysis {

/*!
 * \brief Controls whether parseWaveform writes or accumulates into the
 *        destination buffer.
 */
enum class ParseMode {
    Write,      ///< dst[i] = parsed value (overwrite)
    Accumulate  ///< dst[i] += parsed value (sum into existing)
};

/*!
 * \brief Parse raw digitizer bytes into qint64 values.
 *
 * This is the single implementation of the byte-unpacking loop used by both
 * FtmwConfig::parseWaveform (Write mode) and FtmwScope pre-accumulation
 * (Accumulate mode).
 *
 * The function processes \a numRecords records of \a recordLength samples each,
 * reading \a bytesPerPoint bytes per sample from \a src. Each sample is
 * sign-extended to qint64, multiplied by \a shotMultiplier (to undo firmware
 * block averaging), and left-shifted by \a bitShift (padding bits for peak-up
 * mode).
 *
 * \param src            Pointer to raw waveform bytes.
 * \param dst            Pointer to destination qint64 buffer
 *                       (must hold recordLength * numRecords elements).
 * \param recordLength   Number of samples per record.
 * \param numRecords     Number of records (multi-record mode).
 * \param bytesPerPoint  Bytes per sample (1, 2, or 4).
 * \param byteOrder      Byte order for multi-byte samples.
 * \param shotMultiplier Multiply each sample by this value (1 if no block
 *                       averaging, numAverages otherwise).
 * \param bitShift       Left-shift each sample by this many bits (0 normally,
 *                       8 for peak-up mode).
 * \param mode           Write (overwrite dst) or Accumulate (add to dst).
 */
void parseWaveform(const char *src, qint64 *dst,
                   int recordLength, int numRecords,
                   int bytesPerPoint, DigitizerConfig::ByteOrder byteOrder,
                   quint64 shotMultiplier, quint8 bitShift,
                   ParseMode mode = ParseMode::Write);

void parseBatchParallel(const std::vector<WaveformEntry> &entries,
                        qint64 *dst,
                        int recordLength, int numRecords,
                        int bytesPerPoint, DigitizerConfig::ByteOrder byteOrder,
                        quint64 shotMultiplier, quint8 bitShift);

} // namespace BC::Analysis

#endif // WAVEFORMPARSER_H
