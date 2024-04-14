#ifndef BCGLOBALS_H
#define BCGLOBALS_H

#include <QString>

namespace BC::Key {

static const QString hwIndexSep(".");

QString hwKey(QString k, int index) {
    return QString("%1%2%3").arg(k,hwIndexSep,QString::number(index));
}

std::pair<QString,int> parseKey(const QString key) {
    QStringList l = key.split(hwIndexSep);
    if(l.size() < 2)
        return {key,-1};
    else {
        bool ok = false;
        auto idx = l.at(1).toInt(&ok);
        return {key,ok ? idx : -1};
    }
}

}

namespace BC::Unit{
static const QString us{QString::fromUtf8("μs")};
static const QString MHz("MHz");
static const QString V("V");
static const QString s("s");
static const QString Hz("Hz");
}

#endif // BCGLOBALS_H
