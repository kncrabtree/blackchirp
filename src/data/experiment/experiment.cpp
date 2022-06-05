#include <data/experiment/experiment.h>

#include <data/storage/blackchirpcsv.h>
#include <data/storage/settingsstorage.h>
#include <data/experiment/ftmwconfigtypes.h>

#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/optional/chirpsource/awg.h>
#include <hardware/optional/flowcontroller/flowcontroller.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

#include <QApplication>
#include <QFile>
#include <QDir>
#include <QTextStream>

Experiment::Experiment() : HeaderStorage(BC::Store::Exp::key)
{
    pu_auxData = std::make_unique<AuxDataStorage>();
    pu_validator = std::make_unique<ExperimentValidator>();
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
        case FtmwConfig::LO_Scan:
            pu_ftmwConfig = std::make_unique<FtmwConfigLOScan>(*other.ftmwConfig());
            break;
        case FtmwConfig::DR_Scan:
            pu_ftmwConfig = std::make_unique<FtmwConfigDRScan>(*other.ftmwConfig());
            break;
        default:
            break;
        }
    }

#ifdef BC_LIF
    if(other.lifConfig() != nullptr)
        pu_lifCfg = std::make_unique<LifConfig>(*other.lifConfig());
#endif

    pu_auxData = std::make_unique<AuxDataStorage>(*other.pu_auxData);

    pu_validator = std::make_unique<ExperimentValidator>(*other.pu_validator);

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

    //load hardware list
    QFile hw(d.absoluteFilePath(BC::CSV::hwFile));
    if(hw.open(QIODevice::ReadOnly|QIODevice::Text))
    {
       while(!hw.atEnd())
       {
           auto l = csv->readLine(hw);
           if(l.size() != 2)
               continue;

           auto key = l.constFirst().toString();
           if(key == QString("key"))
               continue;

           d_hardware.insert_or_assign(key,l.constLast().toString());

           //create optional HW configs as needed
           if(key == BC::Key::IOB::ioboard)
               setIOBoardConfig({});

           if(key == BC::Key::PGen::key)
               setPulseGenConfig({});

           if(key == BC::Key::Flow::flowController)
               setFlowConfig({});

           if(key == BC::Key::PController::key)
               setPressureControllerConfig({});

           if(key == BC::Key::TC::key)
               setTempControllerConfig({});

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
            }

#ifdef BC_LIF
            if(key == BC::Config::Exp::lifType)
            {
                enableLif();
                pu_lifCfg->d_number = num;
            }
#endif
        }
    }

    prepareToStore();

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
    if(lifEnabled())
        pu_lifCfg->loadLifData(num,exptPath);
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
    for(auto obj : d_objectives)
    {
        if(!obj->isComplete())
            return false;
    }

    return true;
}

HeaderStorage::HeaderStrings Experiment::getSummary()
{
    auto out = getStrings();

    QString _{""};

    //add hardware information
    for(auto const &[key,val] : d_hardware)
        out.insert({"Hardware",{_,_,key,val,_}});

    return out;
}

void Experiment::backup()
{
    //if we reach this point, it's time to backup
    d_lastBackupTime = QDateTime::currentDateTime();
    pu_ftmwConfig->storage()->backup();
}

FtmwConfig *Experiment::enableFtmw(FtmwConfig::FtmwType type)
{
    if(pu_ftmwConfig.get())
    {
        removeChild(pu_ftmwConfig.get());
        d_objectives.remove(pu_ftmwConfig.get());
        pu_ftmwConfig.reset();
    }

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
    case FtmwConfig::LO_Scan:
        pu_ftmwConfig = std::make_unique<FtmwConfigLOScan>();
        break;
    case FtmwConfig::DR_Scan:
        pu_ftmwConfig = std::make_unique<FtmwConfigDRScan>();
        break;
    default:
        break;
    }

    pu_ftmwConfig->d_type = type;
    d_objectives.insert(pu_ftmwConfig.get());
    return pu_ftmwConfig.get();
}

bool Experiment::initialize()
{
    d_startTime = QDateTime::currentDateTime();
    d_majorVersion = STRINGIFY(BC_MAJOR_VERSION);
    d_minorVersion = STRINGIFY(BC_MINOR_VERSION);
    d_patchVersion = STRINGIFY(BC_PATCH_VERSION);
    d_releaseVersion = STRINGIFY(BC_RELEASE_VERSION);
    d_buildVersion = STRINGIFY(BC_BUILD_VERSION);

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

    d_initSuccess = true;
    return true;

}

void Experiment::abort()
{
    d_isAborted = true;
    d_endLogMessage = QString("Experiment %1 aborted.").arg(d_number);
    d_endLogMessageCode = LogHandler::Error;

    if(ftmwEnabled())
    {
        if(pu_ftmwConfig->d_type == FtmwConfig::Peak_Up)
        {
            d_endLogMessageCode = LogHandler::Highlight;
            d_endLogMessage = QString("Peak up mode ended.");
        }
        if(pu_ftmwConfig->d_type == FtmwConfig::Forever)
        {
            d_endLogMessage = QString("Experiment %1 complete.").arg(d_number);
            d_endLogMessageCode = LogHandler::Highlight;
        }

        pu_ftmwConfig->cleanup();
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
    pu_iobCfg = std::make_unique<IOBoardConfig>(cfg);
}

void Experiment::setPulseGenConfig(const PulseGenConfig &c)
{
    pu_pGenCfg = std::make_unique<PulseGenConfig>(c);
}

void Experiment::setFlowConfig(const FlowConfig &c)
{
    pu_flowCfg = std::make_unique<FlowConfig>(c);
}

void Experiment::setPressureControllerConfig(const PressureControllerConfig &c)
{
    pu_pcConfig = std::make_unique<PressureControllerConfig>(c);
}

void Experiment::setTempControllerConfig(const TemperatureControllerConfig &c)
{
    pu_tcConfig = std::make_unique<TemperatureControllerConfig>(c);
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

#ifdef BC_LIF
LifConfig *Experiment::enableLif()
{
    if(pu_lifCfg.get())
    {
        removeChild(pu_lifCfg.get());
        d_objectives.remove(pu_lifCfg.get());
        pu_lifCfg.reset();
    }

    pu_lifCfg = std::make_unique<LifConfig>();
    d_objectives.insert(pu_lifCfg.get());
    return pu_lifCfg.get();
}
#endif

void Experiment::finalSave()
{
    if(d_isDummy)
        return;

    if(ftmwEnabled())
    {
        pu_ftmwConfig->cleanup();
        pu_ftmwConfig->storage()->save();
    }

#ifdef BC_LIF
    if(lifEnabled())
        lifConfig()->writeLifFile(d_number);
#endif

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
    store(majver,d_majorVersion);
    store(minver,d_minorVersion);
    store(patchver,d_patchVersion);
    store(relver,d_releaseVersion);
    store(buildver,d_buildVersion);
}

void Experiment::retrieveValues()
{
    using namespace BC::Store::Exp;
    d_number = retrieve<int>(num);
    d_timeDataInterval = retrieve<int>(timeData);
    d_backupIntervalHours = retrieve<int>(backupInterval);
    d_majorVersion = retrieve<QString>(majver);
    d_minorVersion = retrieve<QString>(minver);
    d_patchVersion = retrieve<QString>(patchver);
    d_releaseVersion = retrieve<QString>(relver);
    d_buildVersion = retrieve<QString>(buildver);
}

void Experiment::prepareChildren()
{
    addChild(pu_ftmwConfig.get());
    addChild(pu_flowCfg.get());
    addChild(pu_iobCfg.get());
    addChild(pu_pGenCfg.get());
    addChild(pu_pcConfig.get());
    addChild(pu_tcConfig.get());
    addChild(pu_validator.get());
#ifdef BC_LIF
    addChild(pu_lifCfg.get());
#endif

}
