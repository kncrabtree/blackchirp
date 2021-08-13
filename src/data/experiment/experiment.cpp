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

    pu_validator = std::make_unique<ExperimentValidator>(*other.pu_validator);
    addChild(pu_validator.get());

    if(other.pu_iobCfg.get() != nullptr)
        setIOBoardConfig(*other.iobConfig());

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

void Experiment::backup()
{
    //if we reach this point, it's time to backup
    d_lastBackupTime = QDateTime::currentDateTime();
    pu_ftmwConfig->storage()->backup();
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
    addChild(pu_ftmwConfig.get());
    return pu_ftmwConfig.get();
}

bool Experiment::initialize()
{
    d_startTime = QDateTime::currentDateTime();

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
    d_endLogMessage = QString("Experiment %1 aborted.").arg(d_number);
    d_endLogMessageCode = BlackChirp::LogError;

    if(ftmwEnabled())
    {
        if(pu_ftmwConfig->d_type == FtmwConfig::Peak_Up)
        {
            d_endLogMessageCode = BlackChirp::LogHighlight;
            d_endLogMessage = QString("Peak up mode ended.");
        }
        if(pu_ftmwConfig->d_type == FtmwConfig::Forever)
        {
            d_endLogMessage = QString("Experiment %1 complete.").arg(d_number);
            d_endLogMessageCode = BlackChirp::LogHighlight;
        }
    }

}

bool Experiment::canBackup()
{
    if(isComplete() || d_backupIntervalHours < 1 || !d_startTime.isValid())
        return false;

    auto now = QDateTime::currentDateTime();
    if(d_lastBackupTime.isValid())
    {
        if(d_lastBackupTime.addSecs(3600*d_backupIntervalHours) <= now)
            return true;
    }
    else if(d_startTime.addSecs(3600*d_backupIntervalHours) <= now)
        return true;

    return false;
}

void Experiment::setIOBoardConfig(const IOBoardConfig &cfg)
{
    if(pu_iobCfg.get())
        *pu_iobCfg = cfg;
    else
    {
        pu_iobCfg = std::make_unique<IOBoardConfig>(cfg);
        addChild(iobConfig());
    }
}

void Experiment::setPulseGenConfig(const PulseGenConfig &c)
{
    if(pu_pGenCfg.get())
        *pu_pGenCfg = c;
    else
    {
        pu_pGenCfg = std::make_unique<PulseGenConfig>(c);
        addChild(pGenConfig());
    }
}

void Experiment::setFlowConfig(const FlowConfig &c)
{
    if(pu_flowCfg.get())
        *pu_flowCfg = c;
    else
    {
        pu_flowCfg = std::make_unique<FlowConfig>(c);
        addChild(flowConfig());
    }
}

void Experiment::setPressureControllerConfig(const PressureControllerConfig &c)
{
    if(pu_pcConfig.get())
        *pu_pcConfig = c;
    else
    {
        pu_pcConfig = std::make_unique<PressureControllerConfig>(c);
        addChild(pcConfig());
    }
}

void Experiment::setTempControllerConfig(const TemperatureControllerConfig &c)
{
    if(pu_tcConfig.get())
        *pu_tcConfig = c;
    else
    {
        pu_tcConfig = std::make_unique<TemperatureControllerConfig>(c);
        addChild(tcConfig());
    }
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
        pu_ftmwConfig->storage()->save();

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

void Experiment::storeValues()
{
    using namespace BC::Store::Exp;
    store(num,d_number);
    store(timeData,d_timeDataInterval,QString("s"));
    store(backupInterval,d_backupIntervalHours,QString("hr"));
}

void Experiment::retrieveValues()
{
    using namespace BC::Store::Exp;
    d_number = retrieve<int>(num);
    d_timeDataInterval = retrieve<int>(timeData);
    d_backupIntervalHours = retrieve<int>(backupInterval);
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
