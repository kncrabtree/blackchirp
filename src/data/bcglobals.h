#ifndef BCGLOBALS_H
#define BCGLOBALS_H

#include <QString>

namespace BC::Key {

static const QString hwIndexSep(".");

QString hwKey(QString k, int index);

std::pair<QString,int> parseKey(const QString key);

}

namespace BC::Unit{
static const QString us{QString::fromUtf8("μs")};
static const QString MHz("MHz");
static const QString V("V");
static const QString s("s");
static const QString Hz("Hz");
}

#endif // BCGLOBALS_H
