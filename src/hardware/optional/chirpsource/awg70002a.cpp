#include "awg70002a.h"

#include <QtEndian>
#include <math.h>

AWG70002a::AWG70002a(QObject *parent) :
    AWG(BC::Key::AWG::awg70002a,BC::Key::AWG::awg70002aName,CommunicationProtocol::Tcp,parent)
{
    setDefault(BC::Key::AWG::rate,16e9);
    setDefault(BC::Key::AWG::samples,2e9);
    setDefault(BC::Key::AWG::min,100.0);
    setDefault(BC::Key::AWG::max,6250);
    setDefault(BC::Key::AWG::prot,true);
    setDefault(BC::Key::AWG::amp,true);
    setDefault(BC::Key::AWG::rampOnly,false);
    setDefault(BC::Key::AWG::triggered,true);
}


bool AWG70002a::testConnection()
{
    QByteArray resp = p_comm->queryCmd(QString("*IDN?\n"));

    if(resp.isEmpty())
    {
        d_errorString = QString("Did not respond to ID query.");
        return false;
    }

    if(!resp.startsWith(QByteArray("TEKTRONIX,AWG70002A")))
    {
        d_errorString = QString("ID response invalid. Response: %1 (Hex: %2)").arg(QString(resp.trimmed())).arg(QString(resp.toHex()));
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(QString(resp.trimmed())));

    p_comm->writeCmd(QString("*CLS\n"));
    resp = p_comm->queryCmd(QString("System:Error:Count?\n"));
    if(resp.trimmed().toInt() > 0)
        resp = p_comm->queryCmd(QString("System:Error:All?\n"));

    return true;
}

void AWG70002a::initialize()
{
    p_comm->setReadOptions(10000,true,QByteArray("\n"));
}

bool AWG70002a::prepareForExperiment(Experiment &exp)
{
    d_enabledForExperiment = exp.ftmwEnabled();
    if(!d_enabledForExperiment)
        return true;

    //encode error by prepending '!' to an error message
    QString wfmName = getWaveformKey(exp.ftmwConfig()->d_rfConfig.d_chirpConfig);

    if(wfmName.startsWith(QChar('!')))
    {
        exp.setErrorString(wfmName.mid(1));
        emit hardwareFailure();
        return false;
    }

    p_comm->writeCmd(QString("Source1:Waveform \"%1\"\n").arg(wfmName));
    p_comm->writeCmd(QString("Source1:RMode Triggered\n"));
    p_comm->writeCmd(QString("Source1:TINPut ATRigger\n"));
    p_comm->writeCmd(QString("TRIGger:MODE SYNChronous\n"));


    return true;
}

void AWG70002a::beginAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":AWGControl:RUN:Immediate\n"));
        p_comm->queryCmd(QString("*OPC?\n"));
        p_comm->writeCmd(QString(":Output:OFF OFF\n"));
        p_comm->writeCmd(QString(":Output1:State On\n"));
    }

}

void AWG70002a::endAcquisition()
{
    if(d_enabledForExperiment)
    {
        p_comm->writeCmd(QString(":Output:OFF ON\n"));
        p_comm->writeCmd(QString(":AWGControl:STOP:Immediate\n"));
        p_comm->queryCmd(QString("*OPC?\n"));
    }
}

QString AWG70002a::getWaveformKey(const ChirpConfig cc)
{
    //step 1: identify waveform containing chirp; write it if it's not already there
    //encode error by prepending '!' to an error message
    QString wfmHash = QString(cc.waveformHash().toHex());

    QByteArray resp = p_comm->queryCmd(QString("WList:Size?\n"));
    if(resp.isEmpty())
        return QString("!Could not query waveform list size from %1.").arg(d_name);

    bool ok = false;
    int n = resp.trimmed().toInt(&ok);
    if(!ok)
        return QString("Could not parse waveform list size from %1. Response: %2 (Hex: %3)")
                .arg(d_name).arg(QString(resp)).arg(QString(resp.toHex()));

    //get list of known hashes/waveforms
    QList<QPair<QString,QString>> hashList;
    int nEntries = getArraySize(BC::Key::AWG::hashes);
    hashList.reserve(nEntries+1);
    for(int i=0; i<nEntries; i++)
    {
        auto name = getArrayValue(BC::Key::AWG::hashes,i,BC::Key::AWG::wfmName,QString(""));
        auto hash = getArrayValue(BC::Key::AWG::hashes,i,BC::Key::AWG::wfmHash,QString(""));
        if(!name.isEmpty() && !hash.isEmpty())
            hashList.append(qMakePair(name,hash));
    }


    //look up list of waveforms from AWG
    QStringList wfmNames;
    for(int i=0; i<n; i++)
    {
        resp = p_comm->queryCmd(QString("WList:Name? %1\n").arg(i+1));
        if(!resp.isEmpty())
            wfmNames.append(QString(resp.trimmed().replace(QByteArray("\""),QByteArray())));
    }

    //if there were no errors, prune any hashes that are not in list
    if(wfmNames.size() == n)
    {
        for(int i=hashList.size()-1; i>=0; i--)
        {
            if(!wfmNames.contains(hashList.at(i).first))
                hashList.removeAt(i);
        }
    }

    //look for hash match in hash list
    //we know at this point that if the hash matches, the AWG contains the wfm already
    QString nameMatch;
    QString out;
    for(int i=0; i<hashList.size(); i++)
    {
        if(hashList.at(i).second == wfmHash)
            nameMatch = hashList.at(i).first;
    }

    if(nameMatch.isEmpty())
    {
       //write new waveform, get its name, append to hash list
        out = writeWaveform(cc);

        if(out.startsWith(QChar('!')))
            return out;

        hashList.append(qMakePair(out,wfmHash));
    }
    else
        out = nameMatch;

    //write new hash list
    std::vector<SettingsMap> m;
    m.reserve(hashList.size());
    for(auto p : hashList)
        m.push_back({ {BC::Key::AWG::wfmName,p.first}, {BC::Key::AWG::wfmHash,p.second}});
    setArray(BC::Key::AWG::hashes,m);

    return out;
}

QString AWG70002a::writeWaveform(const ChirpConfig cc)
{
    QString name = QDateTime::currentDateTime().toString(QString("yyyy.MM.dd.hh.mm.ss.zzz"));

    QVector<QPointF> data = cc.getChirpMicroseconds();
    QVector<QPair<bool,bool>> markerData = cc.getMarkerData();

    Q_ASSERT(data.size() == markerData.size());

    //create new waveform on AWG
    if(!p_comm->writeCmd(QString("WList:Waveform:New \"%1\", %2\n").arg(name).arg(data.size())))
        return QString("!Could not create new AWG waveform");

    QByteArray resp = p_comm->queryCmd(QString("*OPC?\n"));
    if(resp.isEmpty())
        return QString("!Could not create new AWG waveform. Timed out while waiting for *OPC query");
    if(!resp.startsWith('1'))
        return QString("!Could not create new AWG waveform. *OPC query returned %1 (Hex: %2)")
                .arg(QString(resp)).arg(QString(resp.toHex()));

    //at this point, waveform has been successfully created
    //need to transform waveform data to uint32 LSB first (little endian), and markers to uint8 with bits 6 and 7 set
    //do waveform data first
    //chunks are in bytes
    int chunkSize = 1e6;
    int chunks = static_cast<int>(ceil(static_cast<double>(data.size())*4.0/static_cast<double>(chunkSize)));
    int currentChunk = 0;
    //prepare data
    QByteArray chunkData;
    chunkData.reserve(chunkSize);

    while(currentChunk < chunks)
    {
        chunkData.clear();
        int startIndex = currentChunk*chunkSize/4;
        int endIndex = qMin((currentChunk+1)*chunkSize/4,data.size());
        int numPnts = endIndex - startIndex;

        //downcast double to float
        for(int i=0; i < numPnts; i++)
        {
            float val = static_cast<float>(data.at(startIndex+i).y());
            char *c = reinterpret_cast<char*>(&val);

#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
            chunkData.append(c,4);
#else
            chunkData.append(c[3]).append(c[2]).append(c[1]).append(c[0]);
#endif
        }

        //create data header
        QString header = QString("WList:Waveform:Data \"%1\",%2,%3,")
                .arg(name).arg(startIndex).arg(numPnts);

        QString binSize = QString::number(numPnts*4);
        QString binHeader = QString("#%1%2").arg(binSize.size()).arg(binSize);
        header.append(binHeader);

        if(!p_comm->writeCmd(header))
            return QString("!Could not write header data to AWG. Header: %1").arg(header);

        if(!p_comm->writeBinary(chunkData))
            return QString("!Could not write waveform data to AWG. Header was: %1").arg(header);

        p_comm->writeCmd(QString("\n"));

        resp = p_comm->queryCmd(QString("System:Error:Count?\n"));
        if(resp.trimmed().toInt() > 0)
        {
            resp = p_comm->queryCmd(QString("System:Error:All?\n"));
            emit logMessage(QString("AWG error: %1").arg(QString(resp.trimmed())),LogHandler::Debug);
            return QString("!Could not write waveform data to AWG. See logfile for details. Header was: %1").arg(header);
        }

        currentChunk++;
    }

    //reset for marker data
    currentChunk = 0;
    chunks = static_cast<int>(ceil(static_cast<double>(markerData.size())/static_cast<double>(chunkSize)));
    QByteArray markerChunkData;
    markerChunkData.reserve(chunkSize);

    while(currentChunk < chunks)
    {
        markerChunkData.clear();
        int startIndex = currentChunk*chunkSize;
        int endIndex = qMin((currentChunk+1)*chunkSize,markerData.size());
        int numPnts = endIndex - startIndex;

        for(int i=0; i < numPnts; i++)
        {
            quint8 byte = 0;
            byte += (static_cast<int>(markerData.at(startIndex+i).second) << 7) +
                    (static_cast<int>(markerData.at(startIndex+i).first) << 6);
            markerChunkData.append(byte);
        }


        //create data header
        QString header = QString("WList:Waveform:Marker:Data \"%1\",%2,%3,")
                .arg(name).arg(startIndex).arg(numPnts);

        QString binSize = QString::number(numPnts);
        QString binHeader = QString("#%1%2").arg(binSize.size()).arg(binSize);
        header.append(binHeader);

        if(!p_comm->writeCmd(header))
            return QString("!Could not write header data to AWG. Header: %1").arg(header);

        if(!p_comm->writeBinary(markerChunkData))
            return QString("!Could not write marker data to AWG. Header was: %1").arg(header);

        p_comm->writeCmd(QString("\n"));

        resp = p_comm->queryCmd(QString("System:Error:Count?\n"));
        if(resp.trimmed().toInt() > 0)
        {
            resp = p_comm->queryCmd(QString("System:Error:All?\n"));
            emit logMessage(QString("AWG error: %1").arg(QString(resp.trimmed())),LogHandler::Debug);
             return QString("!Could not write marker data to AWG. See logfile for details. Header was: %1").arg(header);
        }

        currentChunk++;

    }

    return name;
}
