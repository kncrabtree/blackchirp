#include <data/experiment/experiment.h>

#include <data/storage/blackchirpcsv.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/ftmwconfigtypes.h>

#include <QApplication>
#include <QFile>
#include <QDir>
#include <QTextStream>

Experiment::Experiment() : HeaderStorage(BC::Store::Exp::key)
{
    pu_auxData = std::make_unique<AuxDataStorage>();
    pu_validator = std::make_unique<ExperimentValidator>();
    addChild(pu_validator.get());
}

Experiment::Experiment(const Experiment &other) :
    HeaderStorage(BC::Store::Exp::key)
{
    if(other.ftmwConfig() != nullptr)
    {
        switch(other.ftmwConfig()->d_type)
        {
        case FtmwConfig::Target_Shots:
            pu_ftmwConfig = std::make_unique<FtmwConfigSingle>(*other.ftmwConfig());
            break;
        case FtmwConfig::Target_Duration:
            pu_ftmwConfig = std::make_unique<FtmwConfigDuration>(*other.ftmwConfig());
            break;
        case FtmwConfig::Peak_Up:
            pu_ftmwConfig = std::make_unique<FtmwConfigPeakUp>(*other.ftmwConfig());
            break;
        case FtmwConfig::Forever:
            pu_ftmwConfig = std::make_unique<FtmwConfigForever>(*other.ftmwConfig());
            break;
        default:
            break;
        }
    }

    pu_auxData = std::make_unique<AuxDataStorage>(*other.pu_auxData);
    pu_iobCfg = std::make_unique<IOBoardConfig>(*other.pu_iobCfg);
    pu_validator = std::make_unique<ExperimentValidator>(*other.pu_validator);

}

Experiment::Experiment(const int num, QString exptPath, bool headerOnly) : HeaderStorage(BC::Store::Exp::key)
{
    QDir d(BlackchirpCSV::exptDir(num,exptPath));
    if(!d.exists())
        return;

    d_number = num;
    d_path = exptPath;

    //initialize CSV reader
    auto csv = std::make_shared<BlackchirpCSV>(d_number,exptPath);
    pu_validator = std::make_unique<ExperimentValidator>();
    addChild(pu_validator.get());

    //load hardware list
    QFile hw(d.absoluteFilePath(BC::CSV::hwFile));
    if(hw.open(QIODevice::ReadOnly|QIODevice::Text))
    {
       while(!hw.atEnd())
       {
           auto l = csv->readLine(hw);
           if(l.size() != 2)
               continue;

           if(l.constFirst().toString() == QString("key"))
               continue;

           d_hardware.insert_or_assign(l.constFirst().toString(),l.constLast().toString());
       }
    }

    //load objectives
    QFile obj(d.absoluteFilePath(BC::CSV::objectivesFile));
    if(obj.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        while(!obj.atEnd())
        {
            auto l = csv->readLine(obj);

            if(l.size() != 2)
                continue;

            auto key = l.constFirst().toString();
            if(key == QString("key"))
                continue;

            if(key == BC::Config::Exp::ftmwType)
            {
                auto type = l.constLast().value<FtmwConfig::FtmwType>();
                enableFtmw(type);
                pu_ftmwConfig->d_number = num;
                addChild(pu_ftmwConfig.get());
            }

#pragma message("Handle LIF")
        }
    }

    QFile hdr(d.absoluteFilePath(BC::CSV::headerFile));
    if(hdr.open(QIODevice::ReadOnly))
    {
        while(!hdr.atEnd())
        {
            auto l = csv->readLine(hdr);
            if(l.isEmpty())
                continue;

            if(l.constFirst().toString() == QString("ObjKey"))
                continue;

            if(l.size() == 6)
                storeLine(l);
        }
        hdr.close();
        readComplete();
    }
    else
    {
        d_errorString = QString("Could not open header file (%1)").arg(hdr.fileName());
        return;
    }

    if(ftmwEnabled())
    {
        pu_ftmwConfig->d_rfConfig.d_chirpConfig.readChirpFile(csv.get(),num,exptPath);
        pu_ftmwConfig->d_rfConfig.loadClockSteps(csv.get(),num,exptPath);

        if(!headerOnly)
            pu_ftmwConfig->loadFids(num,exptPath);
    }

#ifdef BC_LIF
            if(d_lifCfg.isEnabled())
                d_lifCfg.loadLifData(num,exptPath);
#endif

#ifdef BC_MOTOR
            if(d_motorScan.isEnabled())
                d_motorScan.loadMotorData(num,exptPath);
#endif

    //load aux data
    if(!headerOnly)
        pu_auxData = std::make_unique<AuxDataStorage>(csv.get(),num,exptPath);
    else
        pu_auxData = std::make_unique<AuxDataStorage>();


}

Experiment::~Experiment()
{

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

    ///TODO: Use experiment objective list
    if(ftmwEnabled())
        return pu_ftmwConfig->isComplete();

    return true;
}

QMap<QString, QPair<QVariant, QString> > Experiment::headerMap() const
{
    QMap<QString, QPair<QVariant, QString> > out;

    out.insert(QString("AuxDataInterval"),qMakePair(d_timeDataInterval,QString("s")));
    out.insert(QString("AutosaveInterval"),qMakePair(d_autoSaveIntervalHours,QString("hr")));

//    auto it = d_validationConditions.constBegin();
//    QString prefix("Validation.");
//    QString empty("");
//    for(;it != d_validationConditions.constEnd(); it++)
//    {
//        out.insert(prefix+it.key()+QString(".Min"),qMakePair(it.value().min,empty));
//        out.insert(prefix+it.key()+QString(".Max"),qMakePair(it.value().max,empty));
//    }

//    out.unite(d_ftmwCfg.headerMap());
    out.unite(d_pGenCfg.headerMap());
    out.unite(d_flowCfg.headerMap());
//    out.unite(d_iobCfg.headerMap());

#ifdef BC_LIF
    out.unite(d_lifCfg.headerMap());
#endif

#ifdef BC_MOTOR
    out.unite(d_motorScan.headerMap());
#endif

//    if(!d_timeDataMap.isEmpty())
//    {
//        foreach(const QString &key, d_timeDataMap.keys())
//        {
//            QString label;
//            QString units;

//            QList<QString> flowNames;
//            for(int i=0; i<flowConfig().size(); i++)
//                flowNames.append(flowConfig().setting(i,FlowConfig::Name).toString());

//            if(key.contains(QString("flow.")))
//            {
//                QString channel = key.split(QString(".")).constLast();
//                label = QString("FlowConfigChannel.%1.Average").arg(channel);
//                units = QString("sccm");
//            }
//            else if(key == QString("gasPressure"))
//            {
//                label = QString("FlowConfigPressureAverage");
//                units = QString("kTorr");
//            }
//            else if(flowNames.contains(key))
//            {
//                label = QString("FlowConfigChannel.%1.Average").arg(flowNames.indexOf(key));
//                units = QString("sccm");
//            }
//            else
//                continue;

//            auto val = d_timeDataMap.value(key);
//            if(val.first.isEmpty())
//                continue;

//            if(val.first.constFirst().canConvert(QVariant::Double))
//            {
//                double mean = 0.0;
//                for(int i=0; i<val.first.size(); i++)
//                    mean += val.first.at(i).toDouble();
//                mean /= static_cast<double>(val.first.size());

//                out.insert(label,qMakePair(QString::number(mean,'f',3),units));
//            }

//        }
//    }

    return out;
}

bool Experiment::snapshotReady()
{
    if(isComplete())
        return false;
#pragma message ("Snapshots need work")
//    if(d_ftmwCfg.d_isEnabled)
//    {
//        if(d_ftmwCfg.completedShots() > 0)
//        {
//            qint64 d = d_ftmwCfg.completedShots() - d_lastSnapshot;
//            if(d > 0)
//            {
//                bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
//                if(out)
//                    d_lastSnapshot = d_ftmwCfg.completedShots();
//                return out;
//            }
//            else
//                return false;
//        }
//    }
//#ifdef BC_LIF
//    else if(lifConfig().isEnabled())
//    {
//        if(lifConfig().completedShots() > 0)
//        {
//            qint64 d = lifConfig().completedShots() - d_lastSnapshot;
//            if(d > 0)
//            {
//                bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
//                if(out)
//                    d_lastSnapshot = lifConfig().completedShots();
//                return out;
//            }
//            else
//                return false;
//        }
//    }
//#endif

//#ifdef BC_MOTOR
//    if(motorScan().isEnabled())
//    {
//        if(motorScan().completedShots() > 0)
//        {
//           qint64 d = static_cast<qint64>(motorScan().completedShots()) - d_lastSnapshot;
//           if(d>0)
//           {
//               bool out = !(d % static_cast<qint64>(d_autoSaveShotsInterval));
//               if(out)
//                   d_lastSnapshot = motorScan().completedShots();
//               return out;
//           }
//           else
//               return false;
//        }
//    }
//#endif

    return false;
}

FtmwConfig *Experiment::enableFtmw(FtmwConfig::FtmwType type)
{
    switch(type) {
    case FtmwConfig::Target_Shots:
        pu_ftmwConfig = std::make_unique<FtmwConfigSingle>();
        break;
    case FtmwConfig::Target_Duration:
        pu_ftmwConfig = std::make_unique<FtmwConfigDuration>();
        break;
    case FtmwConfig::Peak_Up:
        pu_ftmwConfig = std::make_unique<FtmwConfigPeakUp>();
        break;
    case FtmwConfig::Forever:
        pu_ftmwConfig = std::make_unique<FtmwConfigForever>();
        break;
    default:
        break;
    }

//    pu_ftmwConfig = std::make_unique<FtmwConfig>();
    pu_ftmwConfig->d_type = type;
    return pu_ftmwConfig.get();
}

bool Experiment::initialize()
{
    d_startTime = QDateTime::currentDateTime();
#pragma message("Add children to initialized experiment")


    int num = -1;


    SettingsStorage s;
    num = s.get(BC::Key::exptNum,0)+1;
    d_number = num;

    if(ftmwEnabled() && pu_ftmwConfig->d_type == FtmwConfig::Peak_Up)
    {
        d_number = -1;
        d_startLogMessage = QString("Peak up mode started.");
        d_endLogMessage = QString("Peak up mode ended.");
        d_isDummy = true;
    }
    else
    {
        d_startLogMessage = QString("Starting experiment %1.").arg(num);
        d_endLogMessage = QString("Experiment %1 complete.").arg(num);
    }

    if(!d_isDummy)
    {
        if(BlackchirpCSV::exptDirExists(num))
        {
            QDir d(BlackchirpCSV::exptDir(num));
            d_errorString = QString("The directory %1 already exists. Update the experiment number or change the experiment path.").arg(d.absolutePath());
            return false;
        }
        if(!BlackchirpCSV::createExptDir(num))
        {
            d_errorString = QString("Could not create experiment directory for saving. Choose a different location.");
            return false;
        }

        //here we've created the directory, so update expt number even if something goes wrong
        //one of the few cases where direct usage of QSettings is needed
        QSettings set{QCoreApplication::organizationName(),QCoreApplication::applicationName()};
        set.setFallbacksEnabled(false);
        set.beginGroup(BC::Key::BC);
        set.setValue(BC::Key::exptNum,num);
        set.endGroup();
        set.sync();
    }

    pu_auxData->d_number = d_number;

    if(ftmwEnabled())
    {
        pu_ftmwConfig->d_number = d_number;
        if(!pu_ftmwConfig->initialize())
        {
            d_errorString = pu_ftmwConfig->d_errorString;
            return false;
        }
        addChild(pu_ftmwConfig.get());
    }

    if(pu_iobCfg)
        addChild(pu_iobCfg.get());

#ifdef BC_LIF
    //do any needed initialization for LIF here... nothing to do for now
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
    {
        d_motorScan.initialize();
    }
#endif


    //write config file, header file; chirps file, and clocks file as appropriate
    if(!d_isDummy)
    {
        if(!BlackchirpCSV::writeVersionFile(d_number))
        {
            d_errorString = QString("Could not open the file %1 for writing.")
                    .arg(BlackchirpCSV::exptDir(d_number).absoluteFilePath(BC::CSV::versionFile));
            return false;
        }

        if(!saveObjectives())
            return false;

        if(!saveHardware())
            return false;

        if(!saveHeader())
            return false;

        //chirp file
        if(ftmwEnabled())
        {
            if(!saveChirpFile())
            {
                d_errorString = QString("Could not open the file %1 for writing.")
                        .arg(BlackchirpCSV::exptDir(d_number).absoluteFilePath(BC::CSV::chirpFile));
                return false;
            }

            if(!saveClockFile())
            {
                d_errorString = QString("Could not open the file %1 for writing.")
                        .arg(BlackchirpCSV::exptDir(d_number).absoluteFilePath(BC::CSV::clockFile));
                return false;
            }
        }
    }

    return true;

}

void Experiment::abort()
{
    d_isAborted = true;
    if(ftmwEnabled() && (pu_ftmwConfig->d_type == FtmwConfig::Target_Shots || pu_ftmwConfig->d_type == FtmwConfig::Target_Duration ))
    {
        d_endLogMessage = QString("Experiment %1 aborted.").arg(d_number);
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

void Experiment::setIOBoardConfig(const IOBoardConfig &cfg)
{
    pu_iobCfg = std::make_unique<IOBoardConfig>(cfg);
}

bool Experiment::addAuxData(AuxDataStorage::AuxDataMap m)
{
    //return false if scan should be aborted
    bool out = true;

    for(auto &[key,val] : m)
    {
        if(!validateItem(key,val))
            break;
    }

    auxData()->addDataPoints(m);

    return out;
}

void Experiment::setValidationMap(const ExperimentValidator::ValidationMap &m)
{
    pu_validator->setValidationMap(m);
}

bool Experiment::validateItem(const QString key, const QVariant val)
{
    bool out = pu_validator->validate(key,val);
    if(!out)
        d_errorString = pu_validator->errorString();

    return out;
}

#pragma message("Finalize Snapshots issue")
//void Experiment::finalizeFtmwSnapshots(const FtmwConfig final)
//{
//    d_ftmwCfg = final;
//    d_ftmwCfg.finalizeSnapshots(d_number,d_path);

//    QFile hdr(BlackChirp::getExptFile(d_number,BlackChirp::HeaderFile,d_path));
//    if(hdr.exists())
//        hdr.copy(hdr.fileName().append(QString(".orig")));
//    saveHeader();

//}

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

bool Experiment::incrementFtmw()
{
    return pu_ftmwConfig->advance();
}

void Experiment::setFtmwClocksReady()
{
    pu_ftmwConfig->hwReady();
}

void Experiment::finalSave()
{
    if(d_isDummy)
        return;

//    //record validation keys
//    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
//    QString keys = s.value(QString("knownValidationKeys"),QString("")).toString();
//    QStringList knownKeyList = keys.split(QChar(';'),QString::SkipEmptyParts);

//    auto it = d_timeDataMap.constBegin();
//    while(it != d_timeDataMap.constEnd())
//    {
//        QString key = it.key();
//        if(!knownKeyList.contains(key))
//            knownKeyList.append(key);
//        it++;
//    }

//    keys.clear();
//    if(knownKeyList.size() > 0)
//    {
//        keys = knownKeyList.at(0);
//        for(int i=1; i<knownKeyList.size();i++)
//            keys.append(QString(";%1").arg(knownKeyList.at(i)));

//        s.setValue(QString("knownValidationKeys"),keys);
//    }

//    saveHeader();

    if(ftmwEnabled())
    {
        auto f = pu_ftmwConfig->storage()->save();
        if(f.isRunning())
            f.waitForFinished();
    }

#ifdef BC_LIF
    if(lifConfig().isEnabled())
            lifConfig().writeLifFile(d_number);
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
        motorScan().writeMotorFile(d_number);
#endif

//    saveTimeFile();
}

bool Experiment::saveObjectives()
{
    QDir d(BlackchirpCSV::exptDir(d_number));
    QFile obj(d.absoluteFilePath(BC::CSV::objectivesFile));
    if(!obj.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        d_errorString = QString("Could not open the file %1 for writing.")
                .arg(d.absoluteFilePath(BC::CSV::objectivesFile));
    }

    QTextStream t(&obj);
    BlackchirpCSV::writeLine(t,{"key","value"});
    if(ftmwEnabled())
        BlackchirpCSV::writeLine(t,{BC::Config::Exp::ftmwType,pu_ftmwConfig->d_type});

    return true;
}

bool Experiment::saveHardware()
{
    QDir d(BlackchirpCSV::exptDir(d_number));
    QFile hw(d.absoluteFilePath(BC::CSV::hwFile));
    if(!hw.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        d_errorString = QString("Could not open the file %1 for writing.")
                .arg(d.absoluteFilePath(BC::CSV::hwFile));
        return false;
    }

    QTextStream t(&hw);
    BlackchirpCSV::writeLine(t,{"key","subKey"});
    for(auto &[key,subKey] : d_hardware)
        BlackchirpCSV::writeLine(t,{key,subKey});

    return true;
}

bool Experiment::saveHeader()
{
    QDir d(BlackchirpCSV::exptDir(d_number));
    QFile hdr(d.absoluteFilePath(BC::CSV::headerFile));
    if(!hdr.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        d_errorString = QString("Could not open the file %1 for writing.")
                .arg(d.absoluteFilePath(BC::CSV::headerFile));
        return false;
    }

    return BlackchirpCSV::writeHeader(hdr,getStrings());
}

bool Experiment::saveChirpFile() const
{
    return pu_ftmwConfig->d_rfConfig.d_chirpConfig.writeChirpFile(d_number);
}

bool Experiment::saveClockFile() const
{
    return pu_ftmwConfig->d_rfConfig.writeClockFile(d_number);
}

void Experiment::snapshot(int snapNum, const Experiment other)
{
    if(d_isDummy)
        return;

//    saveHeader();

    (void)snapNum;
    (void)other;
#pragma message("Implement snapshot")

//    if(ftmwEnabled())
//    {
//        FtmwConfig cf = d_ftmwCfg;
////        cf.storeFids();

//        if(other.number() == d_number && other.isInitialized())
//        {
//            if(cf.subtractFids(other.d_ftmwCfg))
//                cf.writeFids(d_number,d_path,snapNum);
//        }
//        else
//            cf.writeFids(d_number,d_path,snapNum);
//    }

#ifdef BC_LIF
    if(lifConfig().isEnabled())
        lifConfig().writeLifFile(d_number);
#endif

#ifdef BC_MOTOR
    if(motorScan().isEnabled())
        motorScan().writeMotorFile(d_number);
#endif

//    saveTimeFile();
}

void Experiment::saveToSettings() const
{

//    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
//    s.beginGroup(QString("lastExperiment"));

////    s.setValue(QString("ftmwEnabled"),d_ftmwCfg.d_isEnabled);
////    if(d_ftmwCfg.d_isEnabled)
////        d_ftmwCfg.saveToSettings();

//#ifdef BC_LIF
//    s.setValue(QString("lifEnabled"),lifConfig().isEnabled());
//    if(lifConfig().isEnabled())
//        d_lifCfg.saveToSettings();
//#endif

//#ifdef BC_MOTOR
//    s.setValue(QString("motorEnabled"),motorScan().isEnabled());
//    if(motorScan().isEnabled())
//        d_motorScan.saveToSettings();
//#endif

//    s.setValue(QString("autoSaveInterval"),d_autoSaveIntervalHours);
//    s.setValue(QString("auxDataInterval"),d_timeDataInterval);

//    d_iobCfg.saveToSettings();

//    s.remove(QString("validation"));
//    s.beginWriteArray(QString("validation"));
//    int i=0;
//    foreach(BlackChirp::ValidationItem val,d_validationConditions)
//    {
//        s.setArrayIndex(i);
//        s.setValue(QString("key"),val.key);
//        s.setValue(QString("min"),qMin(val.min,val.max));
//        s.setValue(QString("max"),qMax(val.min,val.max));
//        i++;
//    }
//    s.endArray();
//    s.endGroup();

}

//Experiment Experiment::loadFromSettings()
//{
//    Experiment out;

//    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
//    s.beginGroup(QString("lastExperiment"));

////    FtmwConfig f = FtmwConfig::loadFromSettings();
////    if(s.value(QString("ftmwEnabled"),false).toBool())
////        f.setEnabled();
////    out.setFtmwConfig(f);

//#ifdef BC_LIF
//    LifConfig l = LifConfig::loadFromSettings();
//    if(s.value(QString("lifEnabled"),false).toBool())
//        l.setEnabled();

//    out.setLifConfig(l);
//#endif

//#ifdef BC_MOTOR

//    MotorScan m = MotorScan::fromSettings();
//    if(s.value(QString("motorEnabled"),false).toBool())
//        m.setEnabled();

//    out.setMotorScan(m);

//#endif

//    out.setAutoSaveShotsInterval(s.value(QString("autoSaveShots"),10000).toInt());
//    out.setTimeDataInterval(s.value(QString("auxDataInterval"),300).toInt());

//    out.setIOBoardConfig(IOBoardConfig());

//    int num = s.beginReadArray(QString("validation"));
//    for(int i=0; i<num; i++)
//    {
//        s.setArrayIndex(i);
//        bool ok = false;
//        QString key = s.value(QString("key")).toString();
//        double min = s.value(QString("min")).toDouble(&ok);
//        if(ok)
//        {
//            double max = s.value(QString("max")).toDouble(&ok);
//            if(ok && !key.isEmpty())
//                out.addValidationItem(key,min,max);
//        }
//    }
//    s.endArray();
//    s.endGroup();

//    return out;
//}

void Experiment::storeValues()
{
    using namespace BC::Store::Exp;
    store(num,d_number);
    store(timeData,d_timeDataInterval,QString("s"));
    store(autoSave,d_autoSaveIntervalHours,QString("shots"));
    if(pu_ftmwConfig.get() != nullptr)
    {
        store(ftmwEn,true);
        store(ftmwType,pu_ftmwConfig->d_type);
    }
}

void Experiment::retrieveValues()
{
    using namespace BC::Store::Exp;
    d_number = retrieve<int>(num);
    d_timeDataInterval = retrieve<int>(timeData);
    d_autoSaveIntervalHours = retrieve<int>(autoSave);
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
