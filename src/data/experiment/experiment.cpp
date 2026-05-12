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

#include <hardware/core/lifdigitizer/lifdigitizer.h>

#include <QFile>
#include <QSaveFile>
#include <QDir>
#include <QTextStream>

Experiment::Experiment() : HeaderStorage(BC::Store::Exp::key)
{
    ps_auxData = std::make_shared<AuxDataStorage>();
    ps_validator = std::make_shared<ExperimentValidator>();
    ps_overlayStorage = std::make_shared<OverlayStorage>(-1, ""); // Default constructor for new experiments
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

    ps_validator = std::make_shared<ExperimentValidator>();

    //load hardware list using HardwareDataContainer
    d_hardwareData = BC::Data::HardwareDataContainer::loadFromFile(d.absoluteFilePath(BC::CSV::hwFile));
    if (d_hardwareData.hasAnyHardware()) {
        d_hardwareSuccess = true;
        initOptHwFromData();
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

            ExperimentObjective *obj = nullptr;

            if(key == BC::Config::Exp::ftmwType)
            {
                auto type = BC::CSV::enumFromVariant<FtmwConfig::FtmwType>(l.constLast(),FtmwConfig::Forever);
                obj = enableFtmw(type);
            }

            if(key == BC::Config::Exp::lifType)
                obj = enableLif();

            if(obj != nullptr)
            {
                obj->d_number = num;
                obj->d_path = exptPath;
            }

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
            {
                if(storeLine(l))
                    continue;
                else
                {
                    //if line didn't store, maybe this was from a v1.0.0-beta build
                    //that predated the multiple hardware feature, so HW keys did not
                    //have an extension. Append a .0 to make them match a current item
                    auto bVer = csv->buildVersion();
                    if(bVer.startsWith("v0.1"))
                    {
                        auto bl = bVer.split("-");
                        if(bl.size() < 2)
                            continue;
                        auto build = bl.at(1).toInt();
                        if(build >0 && build < 427)
                        {
                            l[0] = l.at(0).toString()+".0";
                            storeLine(l);
                        }
                    }
                }
            }
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
        ps_ftmwConfig->d_rfConfig.d_chirpConfig.readChirpFile(csv.get(),num,exptPath);
        ps_ftmwConfig->d_rfConfig.d_chirpConfig.readMarkersFile(csv.get(),num,exptPath);
        ps_ftmwConfig->d_rfConfig.loadClockSteps(csv.get(),num,exptPath);

        if(!headerOnly)
            ps_ftmwConfig->loadFids();
    }

    if(lifEnabled())
        ps_lifCfg->loadLifData();

    //load aux data
    if(!headerOnly)
        ps_auxData = std::make_shared<AuxDataStorage>(csv.get(),num,exptPath);
    else
        ps_auxData = std::make_shared<AuxDataStorage>();

    //load overlays
    if(!headerOnly)
        ps_overlayStorage = std::make_shared<OverlayStorage>(num, exptPath);
    else
        ps_overlayStorage = std::make_shared<OverlayStorage>(-1, ""); // No overlays in header-only mode


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
    for(auto it = d_hardwareData.hardwareMap.cbegin(); it != d_hardwareData.hardwareMap.cend(); ++it)
        out.insert({"Hardware",{_,_,it.key(),it.value().implementation,_}});

    return out;
}

void Experiment::backup()
{
    d_lastBackupTime = QDateTime::currentDateTime();
    if(!ftmwEnabled())
        return;
    ps_ftmwConfig->storage()->backup();
}

void Experiment::initOptHwFromData()
{
    for (auto it = d_hardwareData.hardwareMap.cbegin(); it != d_hardwareData.hardwareMap.cend(); ++it) {
        const QString& key = it.key();

        switch (it.value().type) {
            case BC::Data::HardwareType::IOBoard:
                addOptHwConfig(IOBoardConfig(key));
                break;
            case BC::Data::HardwareType::PulseGenerator:
                addOptHwConfig(PulseGenConfig(key));
                break;
            case BC::Data::HardwareType::FlowController:
                addOptHwConfig(FlowConfig(key));
                break;
            case BC::Data::HardwareType::PressureController:
                addOptHwConfig(PressureControllerConfig(key));
                break;
            case BC::Data::HardwareType::TemperatureController:
                addOptHwConfig(TemperatureControllerConfig(key));
                break;
            case BC::Data::HardwareType::Unknown:
            case BC::Data::HardwareType::FtmwDigitizer:
            case BC::Data::HardwareType::Clock:
            case BC::Data::HardwareType::AWG:
            case BC::Data::HardwareType::GPIBController:
            case BC::Data::HardwareType::LifDigitizer:
            case BC::Data::HardwareType::LifLaser:
                break;
        }
    }
}

FtmwConfig *Experiment::enableFtmw(FtmwConfig::FtmwType type)
{

    disableFtmw();

    // Find FTMW digitizer key from hardware map
    QString digitizerHwKey;
    for (auto it = d_hardwareData.hardwareMap.cbegin(); it != d_hardwareData.hardwareMap.cend(); ++it) {
        if (it.value().type == BC::Data::HardwareType::FtmwDigitizer) {
            digitizerHwKey = it.key();
            break;
        }
    }

    switch(type) {
    case FtmwConfig::Target_Shots:
        ps_ftmwConfig = std::make_shared<FtmwConfigSingle>(digitizerHwKey);
        break;
    case FtmwConfig::Target_Duration:
        ps_ftmwConfig = std::make_shared<FtmwConfigDuration>(digitizerHwKey);
        break;
    case FtmwConfig::Peak_Up:
        ps_ftmwConfig = std::make_shared<FtmwConfigPeakUp>(digitizerHwKey);
        break;
    case FtmwConfig::Forever:
        ps_ftmwConfig = std::make_shared<FtmwConfigForever>(digitizerHwKey);
        break;
    case FtmwConfig::LO_Scan:
        ps_ftmwConfig = std::make_shared<FtmwConfigLOScan>(digitizerHwKey);
        break;
    case FtmwConfig::DR_Scan:
        ps_ftmwConfig = std::make_shared<FtmwConfigDRScan>(digitizerHwKey);
        break;
    default:
        break;
    }

    ps_ftmwConfig->d_type = type;
    d_objectives.insert(ps_ftmwConfig.get());
    return ps_ftmwConfig.get();
}

void Experiment::disableFtmw()
{
    if(ps_ftmwConfig.get())
    {
        removeChild(ps_ftmwConfig.get());
        d_objectives.remove(ps_ftmwConfig.get());
        ps_ftmwConfig.reset();
    }
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

    if(ftmwEnabled() && ps_ftmwConfig->d_type == FtmwConfig::Peak_Up && !lifEnabled())
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
        QSettings set;
        set.setFallbacksEnabled(false);
        set.beginGroup(BC::Key::BC);
        set.setValue(BC::Key::exptNum,num);
        set.endGroup();
        set.sync();
    }

    ps_auxData->d_number = d_number;

    for(auto obj : d_objectives)
    {
        obj->d_number = d_number;
        if(!obj->initialize())
        {
            d_errorString = obj->d_errorString;
            return false;
        }
    }

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

            if(!saveMarkersFile())
            {
                d_errorString = QString("Could not open the file %1 for writing.")
                        .arg(BlackchirpCSV::exptDir(d_number).absoluteFilePath(BC::CSV::markersFile));
                return false;
            }

            if(!saveClockFile())
            {
                d_errorString = QString("Could not open the file %1 for writing.")
                        .arg(BlackchirpCSV::exptDir(d_number).absoluteFilePath(BC::CSV::clockFile));
                return false;
            }

            //overlays
            ps_overlayStorage = std::make_shared<OverlayStorage>(num,"");
            ps_overlayStorage->save();
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
        if(ps_ftmwConfig->d_type == FtmwConfig::Peak_Up)
        {
            d_endLogMessageCode = LogHandler::Highlight;
            d_endLogMessage = QString("Peak up mode ended.");
        }
        if(ps_ftmwConfig->d_type == FtmwConfig::Forever)
        {
            d_endLogMessage = QString("Experiment %1 complete.").arg(d_number);
            d_endLogMessageCode = LogHandler::Highlight;
        }

    }
    for(auto obj : d_objectives)
        obj->abort();

}

bool Experiment::canBackup()
{
    if(isComplete() || d_backupIntervalMinutes < 1 || !d_startTime.isValid() || !ftmwEnabled())
        return false;

    auto now = QDateTime::currentDateTime();
    if(d_lastBackupTime.isValid())
    {
        if(d_lastBackupTime.addSecs(60*d_backupIntervalMinutes) <= now)
            return true;
    }
    else if(d_startTime.addSecs(60*d_backupIntervalMinutes) <= now)
        return true;

    return false;
}

bool Experiment::addAuxData(AuxDataStorage::AuxDataMap m)
{
    //return false if scan should be aborted
    bool out = true;

    for(auto &[headerStorageKey,val] : m)
    {
        if(!validateItem(headerStorageKey,val))
            break;
    }

    auxData()->addDataPoints(m);

    return out;
}

void Experiment::setValidationMap(const ExperimentValidator::ValidationMap &m)
{
    ps_validator->setValidationMap(m);
}

bool Experiment::validateItem(const QString key, const QVariant val)
{
    bool out = ps_validator->validate(key,val);
    if(!out)
        d_errorString = ps_validator->errorString();

    return out;
}

LifConfig *Experiment::enableLif()
{
    disableLif();

    // Look for LIF digitizer in hardware map using robust type identification
    QString digitizerHwKey;
    for (auto it = d_hardwareData.hardwareMap.cbegin(); it != d_hardwareData.hardwareMap.cend(); ++it) {
        if (it.value().type == BC::Data::HardwareType::LifDigitizer) {
            digitizerHwKey = it.key();
            break;
        }
    }

    ps_lifCfg = std::make_shared<LifConfig>(digitizerHwKey);
    
    // Look for LIF laser to get units information
    for (auto it = d_hardwareData.hardwareMap.cbegin(); it != d_hardwareData.hardwareMap.cend(); ++it) {
        if (it.value().type == BC::Data::HardwareType::LifLaser) {
            // Get laser units from hardware settings
            SettingsStorage s(it.key(), SettingsStorage::Hardware);
            QString units = s.get("units", QString("nm"));
            ps_lifCfg->setLaserUnits(units);
            break;
        }
    }
    
    d_objectives.insert(ps_lifCfg.get());
    return ps_lifCfg.get();
}

void Experiment::disableLif()
{
    if(ps_lifCfg.get())
    {
        removeChild(ps_lifCfg.get());
        d_objectives.remove(ps_lifCfg.get());
        ps_lifCfg.reset();
    }
}

void Experiment::finalSave()
{
    if(d_isDummy)
        return;

    for(auto obj : d_objectives)
        obj->cleanupAndSave();
        
    // Save overlays at experiment completion
    if (ps_overlayStorage)
        ps_overlayStorage->save();
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
    for(auto obj : d_objectives)
        BlackchirpCSV::writeLine(t,{obj->objectiveKey(),obj->objectiveData()});

    return true;
}

bool Experiment::saveHardware()
{
    QDir d(BlackchirpCSV::exptDir(d_number));
    
    // Use HardwareDataContainer's saveToFile method for proper new format serialization
    return d_hardwareData.saveToFile(d.absoluteFilePath(BC::CSV::hwFile));
}

bool Experiment::saveHeader()
{
    QDir d(BlackchirpCSV::exptDir(d_number));
    QSaveFile hdr(d.absoluteFilePath(BC::CSV::headerFile));
    if(!hdr.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        d_errorString = QString("Could not open the file %1 for writing.")
                .arg(d.absoluteFilePath(BC::CSV::headerFile));
        return false;
    }

    BlackchirpCSV::writeHeader(hdr,getStrings());
    return hdr.commit();
}

bool Experiment::saveChirpFile() const
{
    return ps_ftmwConfig->d_rfConfig.d_chirpConfig.writeChirpFile(d_number);
}

bool Experiment::saveMarkersFile() const
{
    return ps_ftmwConfig->d_rfConfig.d_chirpConfig.writeMarkersFile(d_number);
}

bool Experiment::saveClockFile() const
{
    return ps_ftmwConfig->d_rfConfig.writeClockFile(d_number);
}

void Experiment::storeValues()
{
    using namespace BC::Store::Exp;
    store(num,d_number);
    store(timeData,d_timeDataInterval,QString("s"));
    store(backupInterval,d_backupIntervalMinutes,QString("min"));
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
    d_backupIntervalMinutes = retrieve<int>(backupInterval);
    d_majorVersion = retrieve<QString>(majver);
    d_minorVersion = retrieve<QString>(minver);
    d_patchVersion = retrieve<QString>(patchver);
    d_releaseVersion = retrieve<QString>(relver);
    d_buildVersion = retrieve<QString>(buildver);
}

void Experiment::prepareChildren()
{
    addChild(ps_ftmwConfig.get());
    addChild(ps_validator.get());

    for(const auto &[k,p] : d_optHwData)
        addChild(p.get());

    addChild(ps_lifCfg.get());

}
