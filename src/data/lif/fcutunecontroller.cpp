#include <data/lif/fcutunecontroller.h>

#include <algorithm>
#include <cmath>

using namespace Qt::StringLiterals;
using namespace BC::FcuTune;

FcuTuneController::FcuTuneController(QObject *parent) : QObject(parent)
{
    qRegisterMetaType<BC::FcuTune::Result>();
}

FcuTuneController::~FcuTuneController() = default;

bool FcuTuneController::start(const QString &stageKey, double startTrim, const Settings &settings,
                              const LifDigitizerConfig &digiConfig, const LifTrace::LifProcSettings &procSettings)
{
    if(isRunning() || !digiConfig.d_refEnabled)
        return false;

    d_stageKey = stageKey;
    d_startTrim = startTrim;
    pu_sweep = std::make_unique<FcuTuneSweep>(settings,startTrim);
    pu_digiConfig = std::make_unique<LifDigitizerConfig>(digiConfig);
    d_procSettings = procSettings;
    d_result = Result{};
    d_recenters = 0;

    d_state = State::Moving;
    emit progress(0);
    request(pu_sweep->currentTrim());
    return true;
}

void FcuTuneController::abort()
{
    if(d_state != State::Moving && d_state != State::Acquiring)
        return;

    auto r = pu_sweep->result();
    r.status = Status::Aborted;
    r.message = u"Sweep aborted; trim restored to %1."_s.arg(d_startTrim,0,'f',0);
    finish(r,d_startTrim);
}

void FcuTuneController::processWaveform(const QVector<qint8> b)
{
    if(d_state != State::Acquiring)
        return;

    LifTrace t(*pu_digiConfig,b,0,0);
    if(!t.hasRefData())
        return;

    // Saturation check on the reference samples within the gate. Raw
    // samples accumulate over the record's shots (LifTrace divides by the
    // shot count to convert to volts), so compare per-shot values.
    auto full = static_cast<double>((qint64{1} << (8*std::max(1,pu_digiConfig->d_bytesPerPoint) - 1)) - 1);
    auto shots = static_cast<double>(std::max(1,t.shots()));
    auto voltsPerCount = std::abs(t.refYMult());
    auto satV = pu_sweep->settings().saturationVolts;
    auto raw = t.refRaw();
    auto start = std::clamp(d_procSettings.refGateStart,0,static_cast<int>(raw.size())-1);
    auto end = std::clamp(d_procSettings.refGateEnd,start,static_cast<int>(raw.size())-1);
    bool clipped = std::any_of(raw.cbegin()+start,raw.cbegin()+end+1,[=](qint64 v){
        auto perShot = static_cast<double>(v)/shots;
        if(perShot >= full || perShot <= -full-1.0)
            return true;
        return satV > 0.0 && std::abs(perShot)*voltsPerCount >= satV;
    });

    if(!pu_sweep->addWaveform(t.refIntegral(d_procSettings),clipped))
    {
        emit progress(pu_sweep->perMilComplete());
        return;
    }

    auto [mean,err] = pu_sweep->pointStats(pu_sweep->currentIndex());
    emit pointComplete(pu_sweep->currentTrim(),mean,err);
    emit progress(pu_sweep->perMilComplete());

    if(pu_sweep->advance())
    {
        d_state = State::Moving;
        request(pu_sweep->currentTrim());
        return;
    }

    auto r = pu_sweep->result();
    if(r.status == Status::PeakAtEdge && d_recenters < pu_sweep->settings().maxRecenters)
    {
        d_recenters++;
        auto settings = pu_sweep->settings();
        pu_sweep = std::make_unique<FcuTuneSweep>(settings,r.center);
        emit recentered(r.center,d_recenters);
        emit progress(0);
        d_state = State::Moving;
        request(pu_sweep->currentTrim());
        return;
    }

    if(r.success())
    {
        auto center = std::round(r.center);
        r.message = u"Peak at trim %1 ± %2; trim set to %3."_s
                        .arg(r.center,0,'f',0).arg(r.centerUncertainty,0,'f',0).arg(center,0,'f',0);
        finish(r,center);
    }
    else
    {
        r.message += u" Trim restored to %1."_s.arg(d_startTrim,0,'f',0);
        finish(r,d_startTrim);
    }
}

void FcuTuneController::trimUpdated(const QString &stageKey, double trim, int direction, bool success)
{
    Q_UNUSED(direction)
    if(stageKey != d_stageKey)
        return;

    if(success && std::abs(trim - d_pendingTrim) > 0.5)
        return;

    switch(d_state)
    {
    case State::Moving:
        if(success)
        {
            d_state = State::Acquiring;
            return;
        }
        else
        {
            auto r = pu_sweep->result();
            r.status = Status::MoveFailed;
            r.message = u"Could not move to trim %1; trim restored to %2."_s
                            .arg(pu_sweep->currentTrim(),0,'f',0).arg(d_startTrim,0,'f',0);
            finish(r,d_startTrim);
        }
        return;
    case State::Finishing:
        d_state = State::Idle;
        emit finished(d_result,trim,success);
        return;
    default:
        return;
    }
}

void FcuTuneController::finish(Result result, double finalTrim)
{
    d_result = std::move(result);
    d_state = State::Finishing;
    request(finalTrim);
}

void FcuTuneController::request(double trim)
{
    d_pendingTrim = trim;
    emit requestTrim(d_stageKey,trim);
}
