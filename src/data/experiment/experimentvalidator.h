#ifndef EXPERIMENTVALIDATOR_H
#define EXPERIMENTVALIDATOR_H

#include <data/storage/headerstorage.h>

#include <QString>
#include <QVariant>
#include <map>

class BlackchirpCSV;

namespace BC::Store::Validator {
inline constexpr QLatin1StringView key{"Validation"};
inline constexpr QLatin1StringView condition{"Condition"};
inline constexpr QLatin1StringView objKey{"ObjKey"};
inline constexpr QLatin1StringView valKey{"ValKey"};
inline constexpr QLatin1StringView min{"Min"};
inline constexpr QLatin1StringView max{"Max"};
}

class ExperimentValidator : public HeaderStorage
{
public:
    using ValueRange = std::pair<double,double>;
    using ObjectMap = std::map<QString,ValueRange,std::less<>>;
    using ValidationMap = std::map<QString,ObjectMap,std::less<>>;

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
