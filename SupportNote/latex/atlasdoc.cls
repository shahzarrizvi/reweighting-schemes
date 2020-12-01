%-------------------------------------------------------------------------------
% Template class for ATLAS documents
%
% See ChangeLog or Git log for history of changes.
% Responsible: Ian Brock (ian.brock@cern.ch)
%
% Note that this file can be overwritten when atlaslatex is updated.
% if TeX Live 2014 is used and xpatch is needed, have to load mhchem first.
%
% Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration
%-------------------------------------------------------------------------------
\def\ATPackageVersion{09-02-00}

\ProvidesClass{latex/atlasdoc}[2020/10/15 v.09-02-00 ATLAS document class]
% Needs 2013 TeX Live or later
\NeedsTeXFormat{LaTeX2e}[2013/01/01]

%-------------------------------------------------------------------------------
% More advanced if syntax
\RequirePackage{ifthen}
% More advanced command definition
\RequirePackage{xparse}
% Process options. Use kvoptions package.
\RequirePackage{kvoptions}
%\RequirePackage[debugshow]{kvoptions}
\SetupKeyvalOptions{
  family=ADOC,
  prefix=ADOC@
}
\DeclareBoolOption[true]{atlasstyle}
\DeclareBoolOption[true]{atlasdraft}
\DeclareBoolOption[false]{coverpage}
\DeclareBoolOption[false]{cernpreprint}
\DeclareBoolOption[false]{preprint}
\DeclareBoolOption[true]{NOTE}
\DeclareBoolOption[false]{PAPER}
\DeclareBoolOption[false]{CONF}
\DeclareBoolOption[false]{PUB}
\DeclareBoolOption[false]{BOOK}
\DeclareBoolOption[false]{HEPDATA}
\DeclareBoolOption[false]{LANGEDIT}
\DeclareBoolOption[true]{LANGSHOW}
\DeclareStringOption[2016]{texlive}
\DeclareStringOption{paper}
\DeclareStringOption[normal]{titlesize}
\DeclareStringOption[\subsectiontocdepth]{tocdepth}
\DeclareBoolOption[true]{floatopt}
\DeclareBoolOption[true]{mathlines}
\DeclareBoolOption[true]{linenofix}
\DeclareBoolOption[false]{texmf}
\DeclareBoolOption[true]{newtx}
\DeclareBoolOption[false]{txfonts}
\DeclareBoolOption[false]{libertine}
\DeclareBoolOption[false]{palatino}
\DeclareBoolOption[false]{mhchem}
\DeclareBoolOption[true]{abstract}
\DeclareBoolOption[true]{atlaslogo}
\DeclareBoolOption[true]{cernlogo}
\DeclareBoolOption[true]{datetop}
\DeclareBoolOption[false]{showframe}
\DeclareStringOption[0pt]{titleextraheight}
\DeclareStringOption[0pt]{copyrightoffset}
\ProcessKeyvalOptions{ADOC}

% Standard LaTeX options for paper size
%\newif\ifADOC@aivpaper
%\ADOC@aivpapertrue
%\DeclareOption{a4paper}{\PassOptionsToClass{\CurrentOption}{\ADOC@mainclass}} % \ADOC@aivpapertrue} %\def\ADOC@paperstring{a4paper}}
%\DeclareOption{letterpaper}{\PassOptionsToClass{\CurrentOption}{\ADOC@mainclass}} % \ADOC@aivpaperfalse} %\def\ADOC@paperstring{letterpaper}}
%\ProcessOptions\relax
\newcommand*{\ATTeXLiveVersion}{\ADOC@texlive}

\newif\ifADOC@CONFnote
\newif\ifADOC@PUBnote

% CERN preprint or auxiliary material
\ifADOC@cernpreprint
  \ADOC@HEPDATAfalse
  \ADOC@atlasdraftfalse
\else
  \ifADOC@preprint
    \ADOC@cernpreprinttrue
    \ADOC@HEPDATAfalse
  \else
    \ifADOC@HEPDATA
      \ADOC@cernpreprintfalse
    \fi
  \fi
\fi

% Directory for style files
\ifADOC@texmf
  \providecommand*{\ATLASLATEXPATH}{}
\else
  \providecommand*{\ATLASLATEXPATH}{latex/}
\fi

% Type of document
\newcommand*{\ADOC@docheadingdefault}{}
% Turn off backref in atlasbiblatex for CERN preprint
\def\ADOC@atlasbiblatex{\ATLASLATEXPATH atlasbiblatex}
\ifADOC@cernpreprint
  \typeout{atlasdoc: ATLAS CERN preprint}
  \renewcommand*{\ADOC@docheadingdefault}{ATLAS Paper}
  \ADOC@PAPERtrue
  \ADOC@NOTEfalse
  \ADOC@CONFnotefalse
  \ADOC@PUBnotefalse
  \ADOC@BOOKfalse
  \typeout{atlasdoc: turning off backref for a CERN preprint}
  \PassOptionsToPackage{backref=false}{\ADOC@atlasbiblatex}
\else
  \ifADOC@PAPER
    \typeout{atlasdoc: ATLAS paper}
    \renewcommand*{\ADOC@docheadingdefault}{ATLAS Paper}
    \ADOC@PAPERtrue
    \ADOC@NOTEfalse
    \ADOC@CONFnotefalse
    \ADOC@PUBnotefalse
    \ADOC@BOOKfalse
  \else
    \ifADOC@CONF
      \typeout{atlasdoc: ATLAS CONF note}
      \renewcommand*{\ADOC@docheadingdefault}{ATLAS CONF Note}
      \ADOC@PAPERfalse
      \ADOC@NOTEfalse
      \ADOC@CONFnotetrue
      \ADOC@PUBnotefalse
      \ADOC@BOOKfalse
    \else
      \ifADOC@PUB
        \typeout{atlasdoc: ATLAS PUB note}
        \renewcommand*{\ADOC@docheadingdefault}{ATLAS PUB Note}
        \ADOC@PAPERfalse
        \ADOC@NOTEfalse
        \ADOC@CONFnotefalse
        \ADOC@PUBnotetrue
        \ADOC@BOOKfalse
      \else
        \ifADOC@BOOK
          \typeout{atlasdoc: ATLAS BOOK}
          \ADOC@PAPERfalse
          \ADOC@NOTEfalse
          \ADOC@CONFnotefalse
          \ADOC@PUBnotefalse
          \ADOC@BOOKtrue
        \else
          \typeout{atlasdoc: ATLAS Note}
          \renewcommand*{\ADOC@docheadingdefault}{ATLAS Note}
        \fi
      \fi
    \fi
  \fi
\fi

% Decide on the main class
\ifADOC@BOOK
  \typeout{atlasdoc: use KOMA-Script scrbook class}
  \def\ADOC@mainclass{scrbook}
\else
  \typeout{atlasdoc: use KOMA-Script scrartcl class}
  \def\ADOC@mainclass{scrartcl}
\fi

% Set paper size
\ifx\ADOC@paper\@empty
  \def\ADOC@paperstring{paper=a4}
  \PassOptionsToClass{a4paper}{\ADOC@mainclass}
  \typeout{atlasdoc: paper size set to default A4}
\else
  \def\ADOC@paperstring{paper=\ADOC@paper}
  \PassOptionsToClass{\ADOC@paperstring}{\ADOC@mainclass}
  \typeout{atlasdoc: paper size is \ADOC@paper, string for class is \ADOC@paperstring}
\fi

% Option to reduce font sizes on title page
\ifthenelse{\equal{\ADOC@titlesize}{small}}{%
  \newcommand*{\ADOC@maintitlesize}{\LARGE}
  \newcommand*{\ADOC@authorsize}{\large}
  \newcommand*{\ADOC@abstractsize}{\small}
}{%
  \newcommand*{\ADOC@maintitlesize}{\huge}
  \newcommand*{\ADOC@authorsize}{\Large}
  \newcommand*{\ADOC@abstractsize}{\normalsize}
}

% ATLAS draft heading text
\def\ADOC@draftheading{ATLAS DRAFT}
\newcommand*{\AtlasDraftHeading}[1]{\def\ADOC@draftheading{#1}}

% Set up command for authors and ORCID
\NewDocumentCommand \AtlasOrcid { o m } {%
  \IfNoValueTF {#1} {%
    #2%
  }{%
    \href{https://orcid.org/#1}{#2}%
  }%
}

% Switch to txfonts for TeX Live <= 2012
% Need to pass this option to atlaspackage as well
% \ifthenelse{\ADOC@texlive < 2012 \AND \boolean{ADOC@newtx}}{%
%   \typeout{*** atlasdoc: use txfonts instead of newtx}
%   \ADOC@newtxfalse
%   \ADOC@txfontstrue
%   \PassOptionsToPackage{txfonts}{\ATLASLATEXPATH atlaspackage}
% }{%
% }

%-------------------------------------------------------------------------------
% Defaults for title page
% \newdimen\ADOC@skipbeforetitle
% \ADOC@skipbeforetitle=60\p@
% \newcommand{\skipbeforetitle}[1]{\ADOC@skipbeforetitle=#1}

% Any additions here that affect the cover page should also be added to atlascover.sty
\def\@title{Dummy title in atlasdoc\\ Use \textbackslash AtlasTitle to set it}
\newcommand{\AtlasTitleText}{Dummy AtlasTitleText in atlasdoc\\ Use \textbackslash AtlasTitle to set it}
\newcommand{\AtlasTitle}[1]{%
  % \def\@title{#1}%
  \title{#1}%
  \renewcommand{\AtlasTitleText}{#1}%
}
\author{The ATLAS Collaboration}
\date{\today}

% Draft version
\def\ADOC@draftversion{}
\newcommand*{\draftversion}[1]{\def\ADOC@draftversion{#1}}
\newcommand*{\AtlasVersion}[1]{\def\ADOC@draftversion{#1}}

% ATLAS reference code - empty by default
% Also create variable with web page
\def\ADOC@refcode{}
\newcommand*{\AtlasRefCode}[1]{\def\ADOC@refcode{#1}}
\newcommand*{\AtlasPaperWebpage}{https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/\ADOC@refcode}

% ATLAS note number
\def\ADOC@atlasnote{}
\newcommand*{\AtlasNote}[1]{\def\ADOC@atlasnote{#1}}

% ATLAS date - arXiv submission; to be filled in by the Physics Office
\def\ADOC@atlasdate{\today}
\newcommand*{\AtlasDate}[1]{\def\ADOC@atlasdate{#1}}

% Journal
\def\ADOC@journal{}
\newcommand*{\journal}[1]{\def\ADOC@journal{#1}}
\newcommand*{\AtlasJournal}[1]{\def\ADOC@journal{#1}}

% Preprint
\def\ADOC@preprintnumber{}
\newcommand*{\PreprintIdNumber}[1]{\def\ADOC@preprintnumber{#1}}

% Journal
\def\ADOC@journalref{}
\newcommand*{\AtlasJournalRef}[1]{\def\ADOC@journalref{#1}}

% DOI
\def\ADOC@doi{}
\newcommand*{\AtlasDOI}[1]{\def\ADOC@doi{#1}}

% ATLAS document heading
\def\ADOC@docheading{}
\newcommand*{\AtlasHeading}[1]{\def\ADOC@docheading{#1}}

% arXiv
\def\AtlasArXivId{}
\def\AtlasArXivURL{}
\newcommand*{\arXivId}[1]{\def\AtlasArXivId{#1}\def\AtlasArXivURL{http://arXiv.org/abs/#1}}
\newcommand*{\AtlasArXivCat}{[hep-ex]}

% HEP Data
\def\AtlasHepDataRecord{}
\newcommand*{\HepDataRecord}[1]{\def\AtlasHepDataRecord{http://hepdata.cedar.ac.uk/view/ins#1}}

% Submission details
\def\ADOC@subprepend{Journal:}
\newcommand*{\subprepend}[1]{\def\ADOC@subprepend{#1}}

% Abstract text
% \newif\ifADOC@abstract
% \ADOC@abstractfalse
\def\AtlasAbstractText{}
\newcommand{\AtlasAbstract}[1]{\def\AtlasAbstractText{#1}}
\newcommand{\abstracttext}[1]{\def\AtlasAbstractText{#1}}

% Extra macros needs for the cover page - define them here to avoid need for atlasmetadefs.sty
% This also avoids problems wth automatically generated metadata files
\def\ACOV@CommentsDeadline{Comments deadline}
\newcommand{\AtlasCoverCommentsDeadline}[1]{\def\ACOV@CommentsDeadline{#1}}

\def\ACOV@AnalysisTeam{Analysis team}
\newcommand{\AtlasCoverAnalysisTeam}[1]{\def\ACOV@AnalysisTeam{#1}}

\def\ACOV@EgroupEditors{Editor egroup}
\newcommand{\AtlasCoverEgroupEditors}[1]{\def\ACOV@EgroupEditors{#1}}

\def\ACOV@EgroupEdBoard{EdBoard egroup}
\newcommand{\AtlasCoverEgroupEdBoard}[1]{\def\ACOV@EgroupEdBoard{#1}}

% Multiple corresponding editors
\def\ACOV@Editors{} % Corresponding editor name & corresponding.editor@email \\ }
\newcommand{\AtlasCoverEditor}[2]{%
  \g@addto@macro\ACOV@Editors{%
    #1 & (\texttt{#2}) \\
  }%
}

\def\ACOV@EdBoardMembers{}
\newcommand{\AtlasCoverEdBoardMember}[1]{%
  \g@addto@macro\ACOV@EdBoardMembers{%
    #1 \\
  }%
}

\def\ACOV@ReaderMembers{}
\newcommand{\AtlasCoverReaderMember}[1]{%
  \g@addto@macro\ACOV@ReaderMembers{%
    #1 \\
  }%
}

\def\ACOV@LangEditor{language editor contact}
\newcommand{\LEcontact}[1]{\def\ACOV@LangEditor{#1}}

\newif\ifACOV@support
\ACOV@supportfalse
\def\ACOV@TwikiURL{}
%\newcommand{\AtlasCoverTwikiURL}[1]{\def\ACOV@TwikiURL{\url{#1}}}
\newcommand{\AtlasCoverTwikiURL}[1]{\ACOV@supporttrue\def\ACOV@TwikiURL{\url{#1}}}

% Supporting note list
\def\ACOV@Notes{}
\newcommand{\AtlasCoverSupportingNote}[2]{\ACOV@supporttrue%
%  \ifx\ACOV@Notes\@empty
%    \g@addto@macro\ACOV@Notes{\begin{tabular}{ll}}
%  \fi
  \g@addto@macro\ACOV@Notes{%
    #1: \url{#2} \\
  }%
}


%-------------------------------------------------------------------------------
% Invoke main class
% Note that \KOMAoptions is not available in versions of KOMA-Script older than 3.0
\ifADOC@BOOK
  \LoadClass[titlepage,11pt]{scrbook}
\else
  \LoadClass[titlepage,11pt]{scrartcl}
\fi

% Required packages
\RequirePackage{graphicx}
\RequirePackage[svgnames]{xcolor}
\RequirePackage{babel}

% Turn on bold math when using \bfseries
\def\bfseries{\fontseries\bfdefault\selectfont\boldmath}

% Set KOMA-Script options for ATLAS style
\ifADOC@atlasstyle
  \KOMAoptions{headinclude=true, footinclude=false}
  % Put less text on the pages of BOOKs etc.
  \ifADOC@BOOK
    \KOMAoptions{DIV=12, BCOR=5mm, twoside}
  \else
    \KOMAoptions{DIV=14, BCOR=5mm}
  \fi
  \KOMAoptions{pagesize}
  % Do not put a dot after Section/Chapter, nor in the figure caption.
  \KOMAoptions{numbers=noendperiod}
  % Paragraph spacing is half a line
  \KOMAoptions{parskip=half}
  \addtokomafont{disposition}{\rmfamily}
  % Set font size for author. URLs in authors (such as orcid) will be black.
  % \ifthenelse{\ADOC@texlive < 2013}{%
  %   \relax
  % }{%
    \addtokomafont{author}{\ADOC@authorsize\hypersetup{urlcolor=black}}
  % }
  \addtokomafont{part}{\LARGE}
  \addtokomafont{descriptionlabel}{\rmfamily}
  %\setkomafont{pageheadfoot}{\normalsize\sffamily}
  \setkomafont{pagehead}{\normalsize\rmfamily}
  %\setkomafont{publishers}{\normalsize\rmfamily}
  % Set table of contents style to have no dotted lines for subsections
  \ifthenelse{\ADOC@texlive < 2016}{%
    \RequirePackage{tocstyle}
    \usetocstyle{noonewithdot}
  }{%
    \DeclareTOCStyleEntry[beforeskip=0pt plus .2pt,linefill=\hfill]{tocline}{subsection}
    \DeclareTOCStyleEntry[beforeskip=0pt plus .2pt,linefill=\hfill]{tocline}{subsubsection}
  }
  % Table of contents down to \ADOC@tocdepth which is 1 by default
  \setcounter{tocdepth}{\ADOC@tocdepth}
  % Figure and table captions
  \setkomafont{caption}{\normalfont\small}
  \setcapindent{0pt}
  \deffootnote[1em]{1em}{1em}{\textsuperscript{\thefootnotemark}\ }
\fi

% ATLAS draft cover pages for papers, PUB and CONF notes
\def\ADOC@atlascover{\ATLASLATEXPATH atlascover}
\ifADOC@coverpage
  \typeout{atlasdoc: load atlascover package}
  \ifADOC@CONFnote
    \PassOptionsToPackage{CONF,atlasdoc}{\ADOC@atlascover}
  \else
    \ifADOC@PUBnote
      \PassOptionsToPackage{PUB,atlasdoc}{\ADOC@atlascover}
    \else
      \PassOptionsToPackage{atlasdoc}{\ADOC@atlascover}
    \fi
  \fi
  % Pass language editor option to cover pages
  \ifADOC@LANGEDIT
    \PassOptionsToPackage{LANGEDIT}{\ADOC@atlascover}
  \fi
  \RequirePackage{\ADOC@atlascover}
\fi

% Remove the numbers from all the headings (\section, \subsection, etc.)
% \setcounter{secnumdepth}{-1}

% hyperref is needed in many places, so include it here
\RequirePackage{hyperref}

% Font(s)
\ifADOC@txfonts
  % Needs special amsmath code in atlaspackage
  \typeout{atlasdoc: using txfonts}
  \RequirePackage[varg]{txfonts}
  \RequirePackage[T1]{fontenc}
  % Pass eVkern option to atlaspackage for siunitx
  \def\ADOC@atlaspackage{\ATLASLATEXPATH atlaspackage}
  \PassOptionsToPackage{eVkern}{\ADOC@atlaspackage}
\else
  \ifADOC@libertine
    \typeout{atlasdoc: using Libertine fonts}
    \RequirePackage{libertine}
    \RequirePackage[T1]{fontenc}
    % \RequirePackage[libertine, defaultsups]{newtxmath}
    \RequirePackage[libertine]{newtxmath}
    \else
    \ifADOC@palatino
      \typeout{atlasdoc: using Palatino fonts}
      % Palatino for rm and math | Helvetica for ss | Courier for tt
      \RequirePackage{mathpazo} % math & rm
      \linespread{1.05}        % Palatino needs more leading (space between lines)
      \RequirePackage[scaled]{helvet} % ss
      \RequirePackage{courier} % tt
      \normalfont
      \RequirePackage[T1]{fontenc}
    \else
      \typeout{atlasdoc: using newtx fonts package}
      % \ifthenelse{\ADOC@texlive < 2013}{%
      %   \RequirePackage{newtxtext}
      %   \typeout{*** atlasdoc: include amsmath before newtxmath}
      %   \RequirePackage{amsmath}
      % }{%
      % \RequirePackage[defaultsups]{newtxtext}
        \RequirePackage{newtxtext}
      % }
      \RequirePackage{newtxmath}
    \fi
  \fi
\fi

% Adjust float settings to try to avoid problems.
% These follow the UK FAQ - see also P.199-200 of the LaTeX Book
\ifADOC@floatopt
  \renewcommand{\topfraction}{0.85}
  \renewcommand{\bottomfraction}{0.70}
  \renewcommand{\textfraction}{0.15}
  \renewcommand{\floatpagefraction}{0.66}
  \renewcommand{\dbltopfraction}{0.66}
  \renewcommand{\dblfloatpagefraction}{0.66}
  \setcounter{topnumber}{9}
  \setcounter{bottomnumber}{9}
  \setcounter{totalnumber}{20}
\fi

% Make sure page sizes etc are calculated after font selection
\KOMAoptions{DIV=last}

% Add ability to include line numbers
\ifADOC@mathlines
  \RequirePackage[mathlines]{lineno}
\else
  \RequirePackage{lineno}
\fi
% If you have an old lineno or do not have it, you can get it from the latex/other directory
% \RequirePackage{latex/other/lineno}
% \nolinenumbers

% Show frame for checking dimensions etc.
\ifADOC@showframe
  \RequirePackage{showframe}
\fi

% Try to avoid adding linenomath around unknown environments
% Code adapted from https://tex.stackexchange.com/questions/436011/linenomath-printing-extra-numbers-on-last-line-of-multline-align-flalign-envir
\ifADOC@linenofix
  \RequirePackage{xpatch}
  \newcommand*\linenomathpatchAMS[1]{%
    \expandafter\xpretocmd\csname #1\endcsname {\linenomathAMS}{}{}%
    \expandafter\xpretocmd\csname #1*\endcsname{\linenomathAMS}{}{}%
    \expandafter\xapptocmd\csname end#1\endcsname {\endlinenomath}{}{}%
    \expandafter\xapptocmd\csname end#1*\endcsname{\endlinenomath}{}{}%
  }
  \expandafter\ifx\linenomath\linenomathWithnumbers
    \typeout{atlasdoc: add line numbers to equations}
    \let\linenomathAMS\linenomathWithnumbers
    \xpatchcmd\linenomathAMS{\advance\postdisplaypenalty\linenopenalty}{}{\typeout{atlasdoc xpatch: Extra linenumber patch worked}}{\typeout{*** atlasdoc xpatch: Extra linenumber patch failed}}
  \else
    \typeout{atlasdoc: do not add line numbers to equations}
    \let\linenomathAMS\linenomathNonumbers
  \fi
\fi

% Compute the time in hours and minutes; make new variables \timehh and \timemm
\newcount\timehh\newcount\timemm
\timehh=\time
\divide\timehh by 60 \timemm=\time
\count255=\timehh\multiply\count255 by -60 \advance\timemm by \count255
% Now make a string with date and time
\newcommand{\dateandtime}{\today \ --
  \ifnum\timehh<10 0\fi\number\timehh:\ifnum\timemm<10 0\fi\number\timemm}
% \RequirePackage{scrdate,scrtime}

% Draft heading/footer code
\newcommand{\ADOC@DraftHeading}{%
  \ifthenelse{\ADOC@texlive < 2015}{%
    \RequirePackage[headsepline]{scrpage2}
  }{%
    \RequirePackage[headsepline]{scrlayer-scrpage}
  }
  \ihead{}
  \chead[]{\fbox{\normalfont\ADOC@draftheading}}
  \ohead{}
  \ifoot{\dateandtime}
  \cfoot{}
  \ofoot{\thepage}
  \pagestyle{scrheadings}
}

% Include fancier header and footer for HepData material document
\ifADOC@HEPDATA
  \ifthenelse{\ADOC@texlive < 2015}{%
    \RequirePackage[headsepline]{scrpage2}
  }{%
    \RequirePackage[headsepline]{scrlayer-scrpage}
  }
  \chead{ATLAS Collaboration --- arXiv:\AtlasArXivId\ \AtlasArXivCat}
  \pagestyle{scrheadings}
\fi

% BOOK, i.e. book heading/footer code
% \newcommand{\AtlasBookHeading}{%
%   \ifthenelse{\ADOC@texlive < 2015}{%
%     \RequirePackage[headsepline]{scrpage2}
%   }{%
%     \RequirePackage[headsepline]{scrlayer-scrpage}
%   }
%   \ifADOC@atlasdraft
%     \ifx\ADOC@draftversion\@empty%
%       \relax
%     \else
%       \ihead{\ADOC@draftheading}
%       \ifoot{\dateandtime}
%     \fi
%   \fi
%   \pagestyle{scrheadings}
% }

% Pass comment option to atlascomment.sty
\ifADOC@LANGEDIT
  \ifADOC@LANGSHOW
    \typeout{atlasdoc: turn on display of language editor comments}
    \PassOptionsToPackage{comment=true}{\ATLASLATEXPATH atlascomment}
  \else
    \typeout{atlasdoc: turn off display of language editor comments}
    \PassOptionsToPackage{comment=false}{\ATLASLATEXPATH atlascomment}
  \fi
  \RequirePackage{\ATLASLATEXPATH atlascomment}
\fi

% Length needed for parbox/minipage in title page
% \setlength{\titlepageparbox}{\linewidth}
% \setlength{\fboxsep}{0pt}
% \addtolength{\titlepageparbox}{-2\fboxrule}
% \addtolength{\titlepageparbox}{-2\fboxsep}
% \addtolength{\titlepageparbox}{-9pt}
% \typeout{titlepageparbox is \the\titlepageparbox}

%-------------------------------------------------------------------------------
% Title page design
% \showboxdepth=\maxdimen
% \showboxbreadth=\maxdimen
\newlength{\titlepageparbox}%
\newcommand{\ATLAS@TitleHeader}{%
  \setlength{\titlepageparbox}{\textwidth}%
  % \addtolength{\titlepageparbox}{-11pt}%
  % \typeout{ATLAS title titlepageparbox is \the\titlepageparbox, text width \the\textwidth, line width \the\linewidth}%
  \parbox[t][35mm]{\titlepageparbox}{%
    \parbox[b][35mm][t]{0.23\linewidth}{%
      \ifADOC@atlaslogo
        \includegraphics[height=25mm]{AT_atlaslogo_2015}\par
      \else
        \rule{0pt}{25mm}\par
      \fi
      \sffamily\mdseries\fontsize{11}{13.2}\selectfont
      \ifx\ADOC@journal\@empty
        \relax
      \else
        \par\ADOC@subprepend\ \ADOC@journal\par
      \fi
    }%
    \hspace*{\fill}%
    \parbox[b][35mm][c]{0.47\linewidth}{%
      \centering
        {\sffamily\bfseries\fontsize{20}{24}\selectfont
         \ifx\ADOC@docheading\@empty
          \typeout{atlasdoc: using default heading}
          \textsf{\ADOC@docheadingdefault}\par
         \else
           \typeout{atlasdoc: using special heading: \ADOC@docheading}
           \textsf{\ADOC@docheading}\par
         \fi
        }
        \vspace*{2ex}
        % ATLAS note number, if any
        \ifx\ADOC@atlasnote\@empty
          % ATLAS reference code, if any and no note number given
          \ifx\ADOC@refcode\@empty
            \mbox{}%
          \else
            {\Large \ADOC@refcode}\par
          \fi
        \else
          {\Large \ADOC@atlasnote}\par
        \fi
        \vspace*{1.5ex}
        % Date, if any
        \ifADOC@datetop
          \ifx\@date\@empty
            \mbox{}\par
          \else
            \@date\par
          \fi
        \fi
    }%
    \hspace*{\fill}%
    \parbox[b][35mm][t]{0.23\linewidth}{%
      \hspace*{\fill}
      \ifADOC@cernlogo%
        \includegraphics[height=25mm]{AT_cernlogo}\par
      \else
        \rule{0pt}{25mm}\par
      \fi
      \ifADOC@atlasdraft
        \ifx\ADOC@draftversion\@empty
          \relax
        \else
          \normalsize\hspace*{\fill}Draft version \ADOC@draftversion
        \fi
      \fi
    }
  }
}

%-------------------------------------------------------------------------------
% CERN preprint title header
\newcommand{\ATLAS@CERNPreprintTitleHeader}{%
  \setlength{\titlepageparbox}{\textwidth}%
  % \addtolength{\titlepageparbox}{-11pt}%
  % \typeout{CERN preprint titlepageparbox is \the\titlepageparbox, text width \the\textwidth, line width \the\linewidth}%
  \parbox[t]{\titlepageparbox}{%
    \makebox[\linewidth]{\centering%
      \sffamily\mdseries\fontsize{14}{16.8}\selectfont%
      EUROPEAN ORGANISATION FOR NUCLEAR RESEARCH (CERN)
    }\par
    \vspace*{2mm}%
    \parbox[t]{0.45\linewidth}{%
      \ifADOC@atlaslogo%
        \includegraphics[height=25mm]{AT_atlaslogo_2015}\par
      \else
        \rule{0pt}{25mm}\par
      \fi
      \sffamily\mdseries\fontsize{11}{13.2}\selectfont
      \ifx\ADOC@journalref\@empty
        \ifx\ADOC@journal\@empty
          \relax
        \else
          Submitted to: \ADOC@journal\par
        \fi
      \else
          \ADOC@journalref\par
      \fi
      \ifx\ADOC@doi\@empty
        \relax
      \else
        DOI: \href{http://dx.doi.org/\ADOC@doi}{\ADOC@doi}\par
      \fi
    }%
    \hspace*{\fill}%
    \parbox[t]{0.45\linewidth}{%
      \hspace*{\fill}
      \ifADOC@cernlogo%
        \includegraphics[height=25mm]{AT_cernlogo}\par
      \else
        \rule{0pt}{25mm}\par
      \fi
      \sffamily\mdseries\fontsize{11}{13.2}\selectfont
      \hspace*{\fill}\ADOC@preprintnumber\par
      \hspace*{\fill}\ADOC@atlasdate\par
    }%
    \par
    \vspace*{2mm}
    \rule{\linewidth}{1pt}\par
  }%
}

%-------------------------------------------------------------------------------
% Auxiliary material title header
\newcommand{\ATLAS@HepDataTitleHeader}{%
  \setlength{\titlepageparbox}{\textwidth}%
  % \typeout{HepData titlepageparbox is \the\titlepageparbox, text width \the\textwidth, line width \the\linewidth}%
  \parbox[t]{\titlepageparbox}{%
    \makebox[\linewidth]{\centering%
      \sffamily\mdseries\fontsize{14}{16.8}\selectfont%
      EUROPEAN ORGANISATION FOR NUCLEAR RESEARCH (CERN)
    }\par
    \vspace*{2mm}%
    \parbox[t]{0.45\linewidth}{%
      \includegraphics[height=25mm]{AT_atlaslogo_2015}\par
      \sffamily\mdseries\fontsize{11}{13.2}\selectfont
      arXiv: \AtlasArXivId\ \AtlasArXivCat
    }
    \hspace*{\fill}
    \parbox[t]{0.45\linewidth}{%
      \hspace*{\fill}\includegraphics[height=25mm]{AT_cernlogo}\par
      \sffamily\mdseries\fontsize{11}{13.2}\selectfont
      \hspace*{\fill}\ADOC@preprintnumber\par
      \hspace*{\fill}\ADOC@atlasdate\par
    }\par
    \vspace*{2mm}
    \rule{\linewidth}{1pt}\par
  }
}

%-------------------------------------------------------------------------------
% Book title header - only ATLAS logo
\newcommand{\ATLAS@BookTitleHeader}{%
  \setlength{\titlepageparbox}{\textwidth}%
  \parbox[t]{\titlepageparbox}{%
    \includegraphics[height=25mm]{AT_atlaslogo_2015}\par
    \hspace*{\fill}
  }
}

%-------------------------------------------------------------------------------
% Copyright cover footer - parbox does not introduce spurious vertical space
\newcommand{\ATLAS@CopyrightFooter}{%
  \setlength{\titlepageparbox}{\textwidth}
  \addtolength{\titlepageparbox}{-11pt}
  % \typeout{Copyright titlepageparbox is \the\titlepageparbox, text width \the\textwidth, line width \the\linewidth}
  % \parbox[b]{\titlepageparbox}{%
  {%
    \rmfamily\mdseries\fontsize{10}{12}\selectfont
    \ifADOC@atlaslogo%
      \copyright\ \the\year \ CERN for the benefit of the ATLAS Collaboration.\newline
      Reproduction of this article or parts of it is allowed as specified in the CC-BY-4.0 license.
    \else
      \copyright\ \the\year \ CERN\newline
      Reproduction of this article or parts of it is allowed as specified in the CC-BY-4.0 license.
    \fi
  }
}

% Information block for book option
\ifADOC@BOOK
  \newcommand{\ATLAS@BookInfo}{%
    \begin{minipage}[b][0.4\textheight]{\textwidth}
      \large\rmfamily\flushleft
      \begin{tabbing}
        Set first column \= Set second column \kill
        \ifADOC@atlasdraft
          \ifx\ADOC@draftversion\@empty
            Reference: \> \ADOC@preprintnumber\\
              \> \ADOC@refcode\\
            Created:   \> \ADOC@atlasdate \\
            Last modified: \> \today \\
          \else
            Draft version: \> \ADOC@draftversion\\
            Created: \> \dateandtime\\
          \fi
        \else
          Reference: \> \ADOC@preprintnumber\\
            \> \ADOC@refcode\\
          Created:   \> \ADOC@atlasdate \\
          Last modified: \> \today \\
        \fi
        \textbf{Prepared by:} \> \textbf{The ATLAS Collaboration}
      \end{tabbing}\par
    \end{minipage}
  }
\fi


% Redefine \maketitle to use KOMA-Script titlepage environment
% Do it at Begin Document so page dimensions are correct
\renewcommand{\maketitle}{%
  % \typeout{In maketitle: line width1 is \the\linewidth, text width is \the\textwidth}
  \begin{titlepage}
    \setlength{\parindent}{\z@}
    \setlength{\parskip}{\z@}
    % \setlength{\parfillskip}{\z@}
    % \typeout{In maketitle: line width2 is \the\linewidth, text width is \the\textwidth}
    % Cover pages
    \ifADOC@coverpage%
      % Make sure line numbers are off for cover pages
      \nolinenumbers
      \ACOV@PrintAtlasCover
      \clearpage
      \ACOV@PrintAtlasCoverPageTwo
      \clearpage
      % Turn line numbers on for a draft
      \ifADOC@atlasdraft
        \ifx\ADOC@draftversion\@empty%
          \relax
        \else
          \linenumbers
        \fi
      \fi
    \fi
    % Empty page and start numbering from here
    \thispagestyle{empty}
    \pagenumbering{arabic}
    % Header
    \ifADOC@cernpreprint
      \typeout{atlasdoc: using CERN preprint title page}
      \ATLAS@CERNPreprintTitleHeader
    \else
      \ifADOC@HEPDATA
        \typeout{atlasdoc: using HepData Material title page}
        \ATLAS@HepDataTitleHeader
      \else
        \ifADOC@BOOK
          \typeout{atlasdoc: using ATLAS book title page}
          \ATLAS@BookTitleHeader
        \else
          \typeout{atlasdoc: using ATLAS document title page}
          \ATLAS@TitleHeader
        \fi
      \fi
    \fi
    {\parfillskip=0pt \emergencystretch=0.5\textwidth \par}
    \vspace{4em plus2em minus2em}
    \ifADOC@BOOK
      {\usekomafont{title}\ADOC@maintitlesize\flushleft \@title\par%
        \rule{\linewidth}{1.5pt}\par}
      % Extra information for TDR etc.
      \ATLAS@BookInfo\par
    \else
      {\usekomafont{title}\ADOC@maintitlesize\centering\@title\par}
      \vspace{3em plus1em minus1em}
      {\usekomafont{author}\centering\@author\par}
    \fi
    \vspace{2em}
    % Abstract
    \ifADOC@BOOK
      \relax
    \else
      \if\AtlasAbstractText\@empty
        \mbox{}
      \else
        \begin{quote}\ADOC@abstractsize
          \ifADOC@atlasdraft
            \ifx\ADOC@draftversion\@empty%
              \AtlasAbstractText\par
            \else
              \begin{internallinenumbers}
                \AtlasAbstractText
              \end{internallinenumbers}\par
            \fi
          \else
            \AtlasAbstractText\par
          \fi
        \end{quote}
      \fi
      %\vspace*{\stretch{1}}
      % Submission details
      % \ifx\ADOC@journal\@empty
      %   \relax
      % \else
      %   \ifADOC@cernpreprint
      %     \relax
      %   \else
      %     \ifADOC@PAPER
      %       \vskip 2em
      %       \parbox[c]{\linewidth}{
      %         \centering
      %         \ADOC@subprepend\ \textit{\ADOC@journal}
      %       }\par
      %       \vskip 2em
      %     \fi
      %   \fi
      % \fi
    \fi
    \par\vspace*{\stretch{1}}%
    % Copyright
    \ATLAS@CopyrightFooter
    % Extra author information for special author lists.
    \@thanks\let\@thanks\@empty
    % Abstract for a book is on its own page
    \ifADOC@BOOK
      \ifADOC@abstract
        \cleardoublepage
        \vspace*{\fill}
        \begin{quote}
          \ADOC@abstractsize\flushleft\textbf{Abstract}\newline
          \ifADOC@atlasdraft
            \ifx\ADOC@draftversion\@empty%
              \AtlasAbstractText\par
            \else
              \begin{internallinenumbers}
              \AtlasAbstractText
              \end{internallinenumbers}\par
            \fi
          \else
            \AtlasAbstractText\par
          \fi
        \end{quote}
        \vspace*{\fill}
        \cleardoublepage
      \fi
    \fi
  \end{titlepage}
}

\AtBeginDocument{%
  % Make sure backref option is turned off for final version of document
  % Final version means atlasdraft=false or version number is empty
  % \def\ADOC@atlasbiblatex{\ATLASLATEXPATH atlasbiblatex}
  % \ifADOC@atlasdraft
  %   \ifx\ADOC@draftversion\@empty%
  %     \typeout{atlasdoc: turning off backref, as no version number set}
  %     \PassOptionsToPackage{backref=false}{\ADOC@atlasbiblatex}
  %   \fi
  % \else
  %   \typeout{atlasdoc: turning off backref, as atlasdraft=false}
  %   \PassOptionsToPackage{backref=false}{\ADOC@atlasbiblatex}
  % \fi
  % Turn on line numbers and back references for draft documents
  \ifADOC@atlasdraft
    \ifx\ADOC@draftversion\@empty%
      \typeout{*** atlasdoc: this ATLAS draft document does not have a version number}
    \else
      \typeout{atlasdoc: this is an ATLAS draft document}
      \ADOC@DraftHeading
      \linenumbers
      % \@ifpackageloaded{biblatex}{%
      %   % backref=true adds page numbers of citations to bibliography
      %   \typeout{atlasdoc: turning backref on}
      %   \ExecuteBibliographyOptions{backref=true}
      % }{}
    \fi
  \else
    \typeout{atlasdoc: this is the final version of an ATLAS document}
  \fi
  \ifADOC@linenofix
    \typeout{+++ atlasdoc: using lineno package fix}
    \linenomathpatchAMS{align}
    \linenomathpatchAMS{alignat}
    \linenomathpatchAMS{equation}
    \linenomathpatchAMS{flalign}
    \linenomathpatchAMS{gather}
    \linenomathpatchAMS{multline}
  \fi
  % \typeout{atlasdoc sizes:}
  % \typeout{paper height and width \the\paperheight, \the\paperwidth}
  % \typeout{PDF page height and width \the\pdfpageheight, \the\pdfpagewidth}
  % \typeout{text height and width \the\textheight, \the\textwidth}
  % \typeout{line width \the\linewidth}
  % \typeout{margin width and sep \the\marginparwidth, \the\marginparsep}
}

\endinput
