#ifndef EXPERIMENTVALIDATOR_H
#define EXPERIMENTVALIDATOR_H

#include <data/storage/headerstorage.h>

#include <QString>
#include <QVariant>
#include <map>

class BlackchirpCSV;

namespace BC::Store::Validator {
static const QString key("Validation");
static const QString condition("Condition");
static const QString objKey("ObjKey");
static const QString valKey("ValKey");
static const QString min("Min");
static const QString max("Max");
}

class ExperimentValidator : public HeaderStorage
{
public:
    using ValueRange = std::pair<double,double>;
    using ObjectMap = std::map<QString,ValueRange>;
    using ValidationMap = std::map<QString,ObjectMap>;

    ExperimentValidator();

    bool validate(const QString key, const QVariant val);
    QString errorString() const { return d_errorString; }
    void setValidationMap(const ValidationMap &m){ d_valMap = m; }

private:
    ValidationMap d_valMap;
    QString d_errorString;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // EXPERIMENTVALIDATOR_H
