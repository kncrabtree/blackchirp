#include <data/experiment/experiment.h>

#include <QSettings>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QTextStream>

Experiment::Experiment() : d_number(0), d_timeDataInterval(300), d_autoSaveShotsInterval(10000), d_lastSnapshot(0), d_isInitialized(false),
    d_isAborted(false), d_isDummy(false), d_hardwareSuccess(true), d_endLogMessageCode(BlackChirp::LogHighlight),
    d_path(QString(""))
#ifdef BC_LIF
,  d_waitForLifSet(false)
#endif
{

}

Experiment::Experiment(const int num, QString exptPath) : d_number(0), d_timeDataInterval(300), d_autoSaveShotsInterval(10000), d_lastSnapshot(0), d_isInitialized(false),
    d_isAborted(false), d_isDummy(false), d_hardwareSuccess(true), d_endLogMessageCode(BlackChirp::LogHighlight),
    d_path(QString(""))
#ifdef BC_LIF
,  d_waitForLifSet(false)
#endif
{
    QDir d(BlackChirp::getExptDir(num,exptPath));
    if(!d.exists())
        return;

    d_path = exptPath;
    d_iobCfg = IOBoardConfig(false);

    QFile hdr(BlackChirp::getExptFile(num,BlackChirp::HeaderFile,exptPath));
    if(hdr.open(QIODevice::ReadOnly))
    {
        while(!hdr.atEnd())
        {
            QString line = QString(hdr.readLine());
            if(line.isEmpty())
                continue;

            QStringList l = line.split(QString("\t"));
            if(l.size() < 2)
                continue;

            QString key = l.constFirst();
            QVariant val = QVariant(l.at(1));

            if(key.startsWith(QString("Ftmw")) || key.startsWith(QString("RfConfig")))
                d_ftmwCfg.parseLine(key,val);


            if(key.startsWith(QString("Flow")))
                d_flowCfg.parseLine(key,val);

            if(key.startsWith(QString("IOBoardConfig")))
                d_iobCfg.parseLine(key,val);

            if(key.startsWith(QString("PulseGen")))
                d_pGenCfg.parseLine(key,val);

#ifdef BC_LIF
            if(key.startsWith(QString("Lif")))
                d_lifCfg.parseLine(key,val);
#endif

#ifdef BC_MOTOR
            if(key.startsWith(QString("Motor")))
                d_motorScan.parseLine(key,val);
#endif

            if(key.startsWith(QString("AutosaveInterval")))
                d_autoSaveShotsInterval = val.toInt();

            if(key.startsWith(QString("AuxDataInterval")))
                d_timeDataInterval = val.toInt();
        }

        hdr.close();

        if(d_ftmwCfg.isEnabled())
        {
            d_ftmwCfg.loadChirps(num,exptPath);
            d_ftmwCfg.loadFids(num,exptPath);
            d_ftmwCfg.loadClocks(num,exptPath);
        }

#ifdef BC_LIF
        if(d_lifCfg.isEnabled())
            d_lifCfg.loadLifData(num,exptPath);
#endif

#ifdef BC_MOTOR
        if(d_motorScan.isEnabled())
            d_motorScan.loadMotorData(num,exptPath);
#endif
    }
    else
    {
        d_errorString = QString("Could not open header file (%1)").arg(hdr.fileName());
        return;
    }

    //load time data
    QFile tdt(BlackChirp::getExptFile(num,BlackChirp::TimeFile,exptPath));
    if(tdt.open(QIODevice::ReadOnly))
    {
        bool plot = true;
        bool lookForHeader = true;
        QStringList hdrList;

        while(!tdt.atEnd())
        {
            QByteArray line = tdt.readLine().trimmed();

            if(line.isEmpty())
                continue;

            if(line.startsWith('#'))
            {
                if(line.endsWith("NoPlotData"))
                {
                    plot = false;
                    lookForHeader = true;
                    hdrList.clear();
                    continue;
                }
                else if(line.endsWith("PlotData"))
                {
                    plot = true;
                    lookForHeader = true;
                    hdrList.clear();
                    continue;
                }
                else
                    continue;
            }

            QByteArrayList l = line.split('\t');
            if(l.isEmpty())
                continue;

            if(lookForHeader)
            {
                for(int i=0; i<l.size(); i++)
                {
                    QByteArrayList l2 = l.at(i).split('_');
                    QString name;
                    for(int j=0; j<l2.size()-1; j++)
                        name += QString(l2.at(j));

                    hdrList.append(name);
                    d_timeDataMap[name] = qMakePair(QList<QVariant>(),plot);
                }
                lookForHeader = false;
            }
            else
            {
                if(l.size() != hdrList.size())
                    continue;

                for(int i=0; i<l.size(); i++)
                {
                    if(hdrList.at(i).contains(QString("TimeStamp")))
                        d_timeDataMap[hdrList.at(i)].first.append(QDateTime::fromString(l.at(i).trimmed(),Qt::ISODate));
                    else
                        d_timeDataMap[hdrList.at(i)].first.append(QString(l.at(i).trimmed()));
                }
            }

        }

        tdt.close();
    }


    d_number = num;

}

Experiment::~Experiment()
{

}

int Experiment::number() const
{
    return d_number;
}

QDateTime Experiment::startTime() const
{
    return d_startTime;
}

int Experiment::timeDataInterval() const
{
    return d_timeDataInterval;
}

int Experiment::autoSaveShots() const
{
    return d_autoSaveShotsInterval;
}

bool Experiment::isInitialized() const
{
    return d_isInitialized;
}

bool Experiment::isAborted() const
{
    return d_isAborted;
}

bool Experiment::isDummy() const
{
    return d_isDummy;
}

FtmwConfig Experiment::ftmwConfig() const
{
    return d_ftmwCfg;
}

PulseGenConfig Experiment::pGenConfig() const
{
    return d_pGenCfg;
}

FlowConfig Experiment::flowConfig() const
{
    return d_flowCfg;
}

IOBoardConfig Experiment::iobConfig() const
{
    return d_iobCfg;
}

bool Experiment::isComplete() const
{
#ifdef BC_MOTOR
    //if motor scan is enabled, then not possible to do LIF or FTMW
    if(d_motorScan.isEnabled())
        return d_motorScan.isComplete();
#endif

#ifdef BC_LIF
    //check each sub expriment!
    return (d_ftmwCfg.isComplete() && d_lifCfg.isComplete());
#endif

    return d_ftmwCfg.isComplete();
}

bool Experiment::hardwareSuccess() const
{
    return d_hardwareSuccess;
}

QString Experiment::errorString() const
{
    return d_errorString;
}

QMap<QString, QPair<QList<QVariant>, bool> > Experiment::timeDataMap() const
{
    return d_timeDataMap;
}

QString Experiment::startLogMessage() const
{
    return d_startLogMessage;
}

QString Experiment::endLogMessage() const
{
    return d_endLogMessage;
}

BlackChirp::LogMessageCode Experiment::endLogMessageCode() const
{
    return d_endLogMessageCode;
}

QMap<QString, QPair<QVariant, QString> > Experiment::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    out.insert(QString("AuxDataInterval"),qMakePair(d_timeDataInterval,QString("s")));
    out.insert(QString("AutosaveInterval"),qMakePair(d_autoSaveShotsInterval,QString("shots")));

    auto it = d_validationConditions.constBegin();
    QString prefix("Validation.");
    QString empty("");
    for(;it != d_validationConditions.constEnd(); it++)
    {
        out.insert(prefix+it.key()+QString(".Min"),qMakePair(it.value().min,empty));
        out.insert(prefix+it.key()+QString(".Max"),qMakePair(it.value().max,empty));
    }

    out.unite(d_ftmwCfg.headerMap());
    out.unite(d_pGenCfg.headerMap());
    out.unite(d_flowCfg.headerMap());
    out.unite(d_iobCfg.headerMap());

#ifdef BC_LIF
    out.unite(d_lifCfg.headerMap());
#endif

#ifdef BC_MOTOR
    out.unite(d_motorScan.headerMap());
#endif

    if(!d_timeDataMap.isEmpty())
    {
        foreach(const QString &key, d_timeDataMap.keys())
        {
            QString label;
            QString units;

            QList<QString> flowNames;
            for(int i=0; i<flowConfig().size(); i++)
                flowNames.append(flowConfig().setting(i,FlowConfig::Name).toString());

            if(key.contains(QString("flow.")))
            {
                QString channel = key.split(QString(".")).constLast();
                label = QString("FlowConfigChannel.%1.Average").arg(channel);
                units = QString("sccm");
            }
            else if(key == QString("gasPressure"))
            {
                label = QString("FlowConfigPressureAverage");
                units = QString("kTorr");
            }
            else if(flowNames.contains(key))
            {
                label = QString("FlowConfigChannel.%1.Average").arg(flowNames.indexOf(key));
                units = QString("sccm");
            }
            else
                continue;

            auto val = d_timeDataMap.value(key);
            if(val.first.isEmpty())
                continue;

            if(val.first.constFirst().canConvert(QVariant::Double))
            {
                double mean = 0.0;
                for(int i=0; i<val.first.size(); i++)
                    mean += val.first.at(i).toDouble();
                mean /= static_cast<double>(val.first.size());

                out.insert(label,qMakePair(QString::number(mean,'f',3),units));
            }

        }
    }

    return out;
}

QMap<QString, BlackChirp::ValidationItem> Experiment::validationItems() const
{
    return d_validationConditions;
}

bool Experiment::snapshotReady()
{
    if(isComplete())
        return false;

    if(ftmwConfig().isEnabled())
    {
        if(ftmwConfig().completedShots() > 0)
        {
            qint64 d = ftmwConfig().completedShots() - d_lastSnapshot;
            if(d > 0)
            {
                bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
                if(out)
                    d_lastSnapshot = ftmwConfig().completedShots();
                return out;
            }
            else
                return false;
        }
    }
#ifdef BC_LIF
    else if(lifConfig().isEnabled())
    {
        if(lifConfig().completedShots() > 0)
        {
            qint64 d = lifConfig().completedShots() - d_lastSnapshot;
            if(d > 0)
            {
                bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
                if(out)
                    d_lastSnapshot = lifConfig().completedShots();
                return out;
            }
            else
                return false;
        }
    }
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
    {
        if(motorScan().completedShots() > 0)
        {
           qint64 d = static_cast<qint64>(motorScan().completedShots()) - d_lastSnapshot;
           if(d>0)
           {
               bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
               if(out)
                   d_lastSnapshot = motorScan().completedShots();
               return out;
           }
           else
               return false;
        }
    }
#endif

    return false;
}

void Experiment::setTimeDataInterval(const int t)
{
    d_timeDataInterval = t;
}

void Experiment::setAutoSaveShotsInterval(const int s)
{
    d_autoSaveShotsInterval = s;
}

void Experiment::setInitialized()
{
    bool initSuccess = true;
    d_startTime = QDateTime::currentDateTime();

    if(d_ftmwCfg.isEnabled())
    {
        if(!d_ftmwCfg.prepareForAcquisition())
        {
            setErrorString(d_ftmwCfg.errorString());
            d_isInitialized = false;
            return;
        }
    }

#ifdef BC_LIF
    //do any needed initialization for LIF here... nothing to do for now
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
    {
        d_motorScan.initialize();
    }
#endif

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    int num = s.value(QString("exptNum"),0).toInt()+1;
    d_number = num;

    if(ftmwConfig().isEnabled() && ftmwConfig().type() == BlackChirp::FtmwPeakUp)
    {
        d_number = -1;
        d_startLogMessage = QString("Peak up mode started.");
        d_endLogMessage = QString("Peak up mode ended.");
        d_isDummy = true;
        d_isInitialized = true;
        saveToSettings();
        return;
    }
    else
    {
        d_startLogMessage = QString("Starting experiment %1.").arg(num);
        d_endLogMessage = QString("Experiment %1 complete.").arg(num);
    }


    QDir d(BlackChirp::getExptDir(num));
    if(!d.exists())
    {
        initSuccess = d.mkpath(d.absolutePath());
        if(!initSuccess)
        {
            d_isInitialized = false;
            d_errorString = QString("Could not create the directory %1 for saving.").arg(d.absolutePath());
            return;
        }
    }

    //write headers; chirps, etc
    //scan header
    if(!saveHeader())
    {
        d_isInitialized = false;
        d_errorString = QString("Could not open the file %1 for writing.")
                .arg(BlackChirp::getExptFile(d_number,BlackChirp::HeaderFile));
        return;
    }

    //chirp file
    if(d_ftmwCfg.isEnabled())
    {
        if(!saveChirpFile())
        {
            d_isInitialized = false;
            d_errorString = QString("Could not open the file %1 for writing.")
                    .arg(BlackChirp::getExptFile(num,BlackChirp::ChirpFile));
            return;
        }

        if(!saveClockFile())
        {
            d_isInitialized = false;
            d_errorString = QString("Could not open the file %1 for writing.")
                    .arg(BlackChirp::getExptFile(num,BlackChirp::ClockFile));
            return;
        }
    }


    d_isInitialized = initSuccess;



    if(initSuccess)
    {
        s.setValue(QString("exptNum"),number());
        saveToSettings();
    }

}

void Experiment::setAborted()
{
    d_isAborted = true;
    if(ftmwConfig().isEnabled() && (ftmwConfig().type() == BlackChirp::FtmwTargetShots || ftmwConfig().type() == BlackChirp::FtmwTargetTime ))
    {
        d_endLogMessage = QString("Experiment %1 aborted.").arg(number());
        d_endLogMessageCode = BlackChirp::LogError;
    }
#ifdef BC_LIF
    else if(lifConfig().isEnabled() && !lifConfig().isComplete())
    {
        d_endLogMessage = QString("Experiment %1 aborted.").arg(number());
        d_endLogMessageCode = BlackChirp::LogError;
    }
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
    {
        d_endLogMessage = QString("Experiment %1 aborted.").arg(number());
        d_endLogMessageCode = BlackChirp::LogError;
    }
#endif

}

void Experiment::setDummy()
{
    d_isDummy = true;
}

void Experiment::setFtmwConfig(const FtmwConfig cfg)
{
    d_ftmwCfg = cfg;
}

void Experiment::setFtmwEnabled(bool en)
{
    d_ftmwCfg.setEnabled(en);
}

void Experiment::setScopeConfig(const FtmwDigitizerConfig &cfg)
{
    d_ftmwCfg.setScopeConfig(cfg);
}

void Experiment::setRfConfig(const RfConfig cfg)
{
    d_ftmwCfg.setRfConfig(cfg);
}

void Experiment::setIOBoardConfig(const IOBoardConfig cfg)
{
    d_iobCfg = cfg;
}

bool Experiment::setFidsData(const QList<QVector<qint64> > l)
{
    if(!d_ftmwCfg.setFidsData(l))
    {
        setErrorString(ftmwConfig().errorString());
        return false;
    }

    return true;
}

bool Experiment::addFids(const QByteArray newData, int shift)
{
    if(!d_ftmwCfg.addFids(newData,shift))
    {
        setErrorString(ftmwConfig().errorString());
        return false;
    }

    return true;
}

void Experiment::overrideTargetShots(const int target)
{
    d_ftmwCfg.setTargetShots(target);
}

void Experiment::resetFids()
{
    d_ftmwCfg.resetFids();
}

void Experiment::setPulseGenConfig(const PulseGenConfig c)
{
    d_pGenCfg = c;
}

void Experiment::setFlowConfig(const FlowConfig c)
{
    d_flowCfg = c;
}

void Experiment::setErrorString(const QString str)
{
    d_errorString = str;
}

bool Experiment::addTimeData(const QList<QPair<QString, QVariant> > dataList, bool plot)
{
    //return false if scan should be aborted
    bool out = true;
    for(int i=0; i<dataList.size(); i++)
    {
        QString key = dataList.at(i).first;
        QVariant value = dataList.at(i).second;

        if(d_timeDataMap.contains(key))
            d_timeDataMap[key].first.append(value);
        else
        {
            QList<QVariant> newList;
            newList.append(value);
            d_timeDataMap.insert(key,qMakePair(newList,plot));
        }

        if(d_validationConditions.contains(key))
        {
            //convert key if needed
            QString name = BlackChirp::channelNameLookup(key);
            if(name.isEmpty())
                name = key;

            bool ok = false;
            double d = value.toDouble(&ok);

            if(!ok)
            {
                out = false;
                d_errorString = QString("Aborting because the item \"%1\" (value = %2) cannot be converted to a double.").arg(name).arg(value.toString());
                break;
            }
            else
            {
                const BlackChirp::ValidationItem &vi = d_validationConditions.value(key);
                if(d < vi.min || d > vi.max)
                {
                    out = false;
                    d_errorString = QString("Aborting because %1 is outside specified range (Value = %2, Min = %3, Max = %4).")
                            .arg(name).arg(d,0,'g').arg(vi.min,0,'g').arg(vi.max,0,'g');
                    break;
                }
            }
        }
    }

    return out;
}

void Experiment::addTimeStamp()
{
    QString key("exptTimeStamp");
    if(d_timeDataMap.contains(key))
        d_timeDataMap[key].first.append(QDateTime::currentDateTime());
    else
    {
        QList<QVariant> newList;
        newList.append(QDateTime::currentDateTime());
        d_timeDataMap.insert(key,qMakePair(newList,false));
    }
}

void Experiment::setValidationItems(const QMap<QString, BlackChirp::ValidationItem> m)
{
    d_validationConditions = m;
}

void Experiment::addValidationItem(const QString key, const double min, const double max)
{
    BlackChirp::ValidationItem val;
    val.key = key;
    val.min = qMin(min,max);
    val.max = qMax(min,max);
    addValidationItem(val);
}

void Experiment::addValidationItem(const BlackChirp::ValidationItem &i)
{
    d_validationConditions.insert(i.key,i);
}

void Experiment::finalizeFtmwSnapshots(const FtmwConfig final)
{
    d_ftmwCfg = final;
    d_ftmwCfg.finalizeSnapshots(d_number,d_path);

    QFile hdr(BlackChirp::getExptFile(d_number,BlackChirp::HeaderFile,d_path));
    if(hdr.exists())
        hdr.copy(hdr.fileName().append(QString(".orig")));
    saveHeader();

}

#ifdef BC_MOTOR
MotorScan Experiment::motorScan() const
{
    return d_motorScan;
}

void Experiment::setMotorEnabled(bool en)
{
    d_motorScan.setEnabled(en);
}

void Experiment::setMotorScan(const MotorScan s)
{
    d_motorScan = s;
}

bool Experiment::addMotorTrace(const QVector<double> d)
{
    return d_motorScan.addTrace(d);
}
#endif

void Experiment::setHardwareFailed()
{
    d_hardwareSuccess = false;
}

bool Experiment::incrementFtmw()
{
    return d_ftmwCfg.increment();
}

void Experiment::setFtmwClocksReady()
{
    d_ftmwCfg.clocksReady();
}

void Experiment::finalSave() const
{
    if(d_isDummy)
        return;

    //record validation keys
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString keys = s.value(QString("knownValidationKeys"),QString("")).toString();
    QStringList knownKeyList = keys.split(QChar(';'),QString::SkipEmptyParts);

    auto it = d_timeDataMap.constBegin();
    while(it != d_timeDataMap.constEnd())
    {
        QString key = it.key();
        if(!knownKeyList.contains(key))
            knownKeyList.append(key);
        it++;
    }

    keys.clear();
    if(knownKeyList.size() > 0)
    {
        keys = knownKeyList.at(0);
        for(int i=1; i<knownKeyList.size();i++)
            keys.append(QString(";%1").arg(knownKeyList.at(i)));

        s.setValue(QString("knownValidationKeys"),keys);
    }

    saveHeader();

    if(ftmwConfig().isEnabled())
        ftmwConfig().writeFids(d_number);

#ifdef BC_LIF
    if(lifConfig().isEnabled())
            lifConfig().writeLifFile(d_number);
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
        motorScan().writeMotorFile(d_number);
#endif

    saveTimeFile();
}

bool Experiment::saveHeader() const
{
    QFile hdr(BlackChirp::getExptFile(d_number,BlackChirp::HeaderFile));
    if(hdr.open(QIODevice::WriteOnly))
    {
        QTextStream t(&hdr);
        t << BlackChirp::headerMapToString(headerMap());
        t.flush();
        hdr.close();
        return true;
    }
    else
        return false;
}

bool Experiment::saveChirpFile() const
{
    QFile chp(BlackChirp::getExptFile(d_number,BlackChirp::ChirpFile));
    if(chp.open(QIODevice::WriteOnly))
    {
        QTextStream t(&chp);
        t << d_ftmwCfg.chirpConfig().toString();
        t.flush();
        chp.close();
        return true;
    }
    else
        return false;
}

bool Experiment::saveClockFile() const
{
    QFile rfc(BlackChirp::getExptFile(d_number,BlackChirp::ClockFile));
    QString txt = d_ftmwCfg.rfConfig().clockStepsString();
    if(txt.isEmpty())
        return true;

    if(rfc.open(QIODevice::WriteOnly))
    {
        QTextStream t(&rfc);
        t << txt;
        t.flush();
        rfc.close();
        return true;
    }

    return false;
}

bool Experiment::saveTimeFile() const
{
    if(d_timeDataMap.isEmpty())
        return true;

    QFile tdt(BlackChirp::getExptFile(d_number,BlackChirp::TimeFile));
    if(tdt.open(QIODevice::WriteOnly))
    {
        QString tab("\t");
        QString nl("\n");

        QTextStream t(&tdt);

        auto it = d_timeDataMap.constBegin();
        for(;it != d_timeDataMap.constEnd(); it++)
        {
            QString alias = BlackChirp::channelNameLookup(it.key());
            if(!alias.isEmpty())
                t << QString("#Alias") << tab << alias << tab << it.key() << nl;

        }

        t << QString("\n\n");
        t << timeDataText();
        t.flush();
        tdt.close();
        return true;
    }
    else
        return false;
}

QString Experiment::timeDataText() const
{
    QString out;
    QList<QPair<QString,QList<QVariant>>> plot, noPlot;
    QTextStream t(&out);
    t.setRealNumberNotation(QTextStream::ScientificNotation);
    t.setRealNumberPrecision(6);
    QString tab("\t");
    QString nl("\n");

    auto it = d_timeDataMap.constBegin();
    int maxPlotSize = 0, maxNoPlotSize = 0;
    for(;it != d_timeDataMap.constEnd(); it++)
    {
        bool p = it.value().second;
        if(p)
        {
            plot.append(qMakePair(it.key(),it.value().first));
            maxPlotSize = qMax(it.value().first.size(),maxPlotSize);
        }
        else
        {
            noPlot.append(qMakePair(it.key(),it.value().first));
            maxNoPlotSize = qMax(it.value().first.size(),maxNoPlotSize);
        }
    }


    if(!plot.isEmpty())
    {
        t << QString("#PlotData\n\n");
        QString name = plot.constFirst().first;

        t << name << QString("_%1").arg(d_number);
        for(int i=1; i<plot.size(); i++)
        {
            name = plot.at(i).first;
            t << tab << name << QString("_%1").arg(d_number);
        }

        for(int i=0; i<maxPlotSize; i++)
        {
            t << nl;

            if(i >= plot.constFirst().second.size())
                t << QString("NaN");
            else
            {
                if(plot.constFirst().second.at(i).canConvert(QVariant::Double))
                    t << plot.constFirst().second.at(i).toDouble();
                else
                    t << plot.constFirst().second.at(i).toString();
            }

            for(int j=1; j<plot.size(); j++)
            {
                if(i >= plot.at(j).second.size())
                    t << tab << QString("NaN");
                else
                {
                    if(plot.at(j).second.at(i).canConvert(QVariant::Double))
                        t << tab << plot.at(j).second.at(i).toDouble();
                    else
                        t << tab << plot.at(j).second.at(i).toString();
                }
            }
        }
    }

    if(!noPlot.isEmpty())
    {
        t << QString("\n\n#NoPlotData\n\n");
        QString name = noPlot.constFirst().first;

        t << name << QString("_%1").arg(d_number);
        for(int i=1; i<noPlot.size(); i++)
        {
            name = noPlot.at(i).first;
            t << tab <<name << QString("_%1").arg(d_number);
        }

        for(int i=0; i<maxNoPlotSize; i++)
        {
            t << nl;

            if(i >= noPlot.constFirst().second.size())
                t << QString("NaN");
            else
            {
                if(noPlot.constFirst().second.at(i).canConvert(QVariant::Double))
                    t << noPlot.constFirst().second.at(i).toDouble();
                else
                    t << noPlot.constFirst().second.at(i).toString();
            }

            for(int j=1; j<noPlot.size(); j++)
            {
                if(i >= noPlot.at(j).second.size())
                    t << tab << QString("NaN");
                else
                {
                    if(noPlot.at(j).second.at(i).canConvert(QVariant::Double))
                        t << tab << noPlot.at(j).second.at(i).toDouble();
                    else
                        t << tab <<noPlot.at(j).second.at(i).toString();
                }
            }
        }
    }
    t.flush();
    return out;
}

void Experiment::snapshot(int snapNum, const Experiment other) const
{
    if(d_isDummy)
        return;

    saveHeader();

    if(ftmwConfig().isEnabled())
    {
        FtmwConfig cf = ftmwConfig();
        cf.storeFids();

        if(other.number() == d_number && other.isInitialized())
        {
            if(cf.subtractFids(other.ftmwConfig()))
                cf.writeFids(d_number,d_path,snapNum);
        }
        else
            cf.writeFids(d_number,d_path,snapNum);
    }

#ifdef BC_LIF
    if(lifConfig().isEnabled())
        lifConfig().writeLifFile(d_number);
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
        motorScan().writeMotorFile(d_number);
#endif

    saveTimeFile();
}

void Experiment::saveToSettings() const
{

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastExperiment"));

    s.setValue(QString("ftmwEnabled"),ftmwConfig().isEnabled());
    if(ftmwConfig().isEnabled())
        d_ftmwCfg.saveToSettings();

#ifdef BC_LIF
    s.setValue(QString("lifEnabled"),lifConfig().isEnabled());
    if(lifConfig().isEnabled())
        d_lifCfg.saveToSettings();
#endif

#ifdef BC_MOTOR
    s.setValue(QString("motorEnabled"),motorScan().isEnabled());
    if(motorScan().isEnabled())
        d_motorScan.saveToSettings();
#endif

    s.setValue(QString("autoSaveShots"),autoSaveShots());
    s.setValue(QString("auxDataInterval"),timeDataInterval());

    d_iobCfg.saveToSettings();

    s.remove(QString("validation"));
    s.beginWriteArray(QString("validation"));
    int i=0;
    foreach(BlackChirp::ValidationItem val,d_validationConditions)
    {
        s.setArrayIndex(i);
        s.setValue(QString("key"),val.key);
        s.setValue(QString("min"),qMin(val.min,val.max));
        s.setValue(QString("max"),qMax(val.min,val.max));
        i++;
    }
    s.endArray();
    s.endGroup();

}

Experiment Experiment::loadFromSettings()
{
    Experiment out;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(QString("lastExperiment"));

    FtmwConfig f = FtmwConfig::loadFromSettings();
    if(s.value(QString("ftmwEnabled"),false).toBool())
        f.setEnabled();
    out.setFtmwConfig(f);

#ifdef BC_LIF   
    LifConfig l = LifConfig::loadFromSettings();
    if(s.value(QString("lifEnabled"),false).toBool())
        l.setEnabled();

    out.setLifConfig(l);
#endif

#ifdef BC_MOTOR

    MotorScan m = MotorScan::fromSettings();
    if(s.value(QString("motorEnabled"),false).toBool())
        m.setEnabled();

    out.setMotorScan(m);

#endif

    out.setAutoSaveShotsInterval(s.value(QString("autoSaveShots"),10000).toInt());
    out.setTimeDataInterval(s.value(QString("auxDataInterval"),300).toInt());

    out.setIOBoardConfig(IOBoardConfig());

    int num = s.beginReadArray(QString("validation"));
    for(int i=0; i<num; i++)
    {
        s.setArrayIndex(i);
        bool ok = false;
        QString key = s.value(QString("key")).toString();
        double min = s.value(QString("min")).toDouble(&ok);
        if(ok)
        {
            double max = s.value(QString("max")).toDouble(&ok);
            if(ok && !key.isEmpty())
                out.addValidationItem(key,min,max);
        }
    }
    s.endArray();
    s.endGroup();

    return out;
}


#ifdef BC_LIF
bool Experiment::isLifWaiting() const
{
    return d_waitForLifSet;
}

LifConfig Experiment::lifConfig() const
{
    return d_lifCfg;
}

void Experiment::setLifEnabled(bool en)
{
    d_lifCfg.setEnabled(en);
}

void Experiment::setLifConfig(const LifConfig cfg)
{
    d_lifCfg = cfg;
}

bool Experiment::addLifWaveform(const LifTrace t)
{
    return d_lifCfg.addWaveform(t);
}

void Experiment::setLifWaiting(bool wait)
{
    d_waitForLifSet = wait;
}

#endif
