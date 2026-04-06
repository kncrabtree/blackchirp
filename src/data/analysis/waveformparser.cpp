#include <data/analysis/waveformparser.h>

#include <QtEndian>
#include <QThread>
#include <QtConcurrent/QtConcurrent>

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

void parseBatchParallel(const std::vector<WaveformEntry> &entries,
                        qint64 *dst,
                        int recordLength, int numRecords,
                        int bytesPerPoint, DigitizerConfig::ByteOrder byteOrder,
                        quint64 shotMultiplier, quint8 bitShift)
{
    if(entries.empty())
        return;

    const qint64 L = static_cast<qint64>(recordLength) * numRecords;

    auto parseSample = [&](const WaveformEntry &e, qint64 k) -> qint64 {
        if(e.preAccumulated)
            return reinterpret_cast<const qint64*>(e.data.constData())[k];

        qint64 dat = 0;
        if(bytesPerPoint == 1)
        {
            char y = e.data.constData()[k];
            dat = static_cast<qint64>(y);
        }
        else if(bytesPerPoint == 2)
        {
            auto y1 = static_cast<quint8>(e.data.constData()[2*k]);
            auto y2 = static_cast<quint8>(e.data.constData()[2*k + 1]);
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
            auto y1 = static_cast<quint8>(e.data.constData()[4*k]);
            auto y2 = static_cast<quint8>(e.data.constData()[4*k + 1]);
            auto y3 = static_cast<quint8>(e.data.constData()[4*k + 2]);
            auto y4 = static_cast<quint8>(e.data.constData()[4*k + 3]);
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
        return dat;
    };

    constexpr int kMinChunkSamples = 8192;
    int P = qBound(1, static_cast<int>(L / kMinChunkSamples), QThread::idealThreadCount());

    auto processChunk = [&](int chunkIdx) {
        qint64 chunkSize = (L + P - 1) / P;
        qint64 start = chunkIdx * chunkSize;
        qint64 end = qMin(start + chunkSize, L);

        for(qint64 k = start; k < end; ++k)
        {
            dst[k] = parseSample(entries[0], k);
            for(std::size_t n = 1; n < entries.size(); ++n)
                dst[k] += parseSample(entries[n], k);
        }
    };

    if(P == 1)
    {
        processChunk(0);
    }
    else
    {
        QVector<int> chunks(P);
        std::iota(chunks.begin(), chunks.end(), 0);
        QtConcurrent::blockingMap(chunks, [&](int chunkIdx) {
            processChunk(chunkIdx);
        });
    }
}

} // namespace BC::Analysis
