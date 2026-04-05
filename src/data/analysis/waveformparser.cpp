#include <data/analysis/waveformparser.h>

#include <QtEndian>

namespace BC::Analysis {

void parseWaveform(const char *src, qint64 *dst,
                   int recordLength, int numRecords,
                   int bytesPerPoint, DigitizerConfig::ByteOrder byteOrder,
                   quint64 shotMultiplier, quint8 bitShift,
                   ParseMode mode)
{
    for(int j = 0; j < numRecords; ++j)
    {
        for(int i = 0; i < recordLength; ++i)
        {
            qint64 dat = 0;
            int idx = j * recordLength + i;

            if(bytesPerPoint == 1)
            {
                char y = src[idx];
                dat = static_cast<qint64>(y);
            }
            else if(bytesPerPoint == 2)
            {
                auto y1 = static_cast<quint8>(src[2*idx]);
                auto y2 = static_cast<quint8>(src[2*idx + 1]);

                qint16 y = 0;
                y |= y1;
                y |= (y2 << 8);

                if(byteOrder == DigitizerConfig::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = static_cast<qint64>(y);
            }
            else
            {
                auto y1 = static_cast<quint8>(src[4*idx]);
                auto y2 = static_cast<quint8>(src[4*idx + 1]);
                auto y3 = static_cast<quint8>(src[4*idx + 2]);
                auto y4 = static_cast<quint8>(src[4*idx + 3]);

                qint32 y = 0;
                y |= y1;
                y |= (y2 << 8);
                y |= (y3 << 16);
                y |= (y4 << 24);

                if(byteOrder == DigitizerConfig::BigEndian)
                    y = qFromBigEndian(y);
                else
                    y = qFromLittleEndian(y);

                dat = static_cast<qint64>(y);
            }

            if(shotMultiplier > 1)
                dat *= shotMultiplier;

            dat = dat << bitShift;

            if(mode == ParseMode::Write)
                dst[idx] = dat;
            else
                dst[idx] += dat;
        }
    }
}

} // namespace BC::Analysis
