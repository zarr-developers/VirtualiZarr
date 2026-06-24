!!! note
    This Core Team Member Guide was adapted from [Xarray's code team guide](https://github.com/pydata/xarray/blob/main/CORE_TEAM_GUIDE.md), [napari project's Core Developer Guide](https://napari.org/stable/developers/core_dev_guide.html), and the [Pandas maintainers guide](https://pandas.pydata.org/docs/development/maintaining.html).

# Core Team Member Guide

Welcome, new core team member!  We appreciate the quality of your work, and enjoy working with you!
Thank you for your numerous contributions to the project so far.

By accepting the invitation to become a core team member you are **not required to commit to doing any more work** -
VirtualiZarr is a volunteer project, and we value the contributions you have made already.

You can see a list of all the current core team members on our
[@zarr-developers/virtualizarr](https://github.com/orgs/zarr-developers/teams/virtualizarr-core-devs)
GitHub team. Once accepted, you should now be on that list too.
This document offers guidelines for your new role.

## Tasks

VirtualiZarr values a wide range of contributions, only some of which involve writing code.
As such, we do not currently make a distinction between a "core team member", "core developer", "maintainer",
or "triage team member" as some projects do (e.g. [pandas](https://pandas.pydata.org/docs/development/maintaining.html)).
That said, if you prefer to refer to your role as one of the other titles above then that is fine by us!

VirtualiZarr is mostly a volunteer project, so these tasks shouldn’t be read as “expectations”.
**There are no strict expectations**, other than to adhere to our [Code of Conduct](https://github.com/zarr-developers/virtualizarr?tab=coc-ov-file#readme).
Rather, the tasks that follow are general descriptions of what it might mean to be a core team member:

- Facilitate a welcoming environment for those who file issues, make pull requests, and open discussion topics,
- Triage newly filed issues,
- Review newly opened pull requests,
- Respond to updates on existing issues and pull requests,
- Drive discussion and decisions on stalled issues and pull requests,
- Provide experience / wisdom on API design questions to ensure consistency and maintainability,
- Project organization (run developer meetings, coordinate with sponsors),
- Project evangelism (advertise VirtualiZarr to new users),
- Community contact (represent VirtualiZarr in user communities such as [Pangeo](https://pangeo.io/)),
- Key project contact (represent VirtualiZarr's perspective within key related projects like Xarray, Zarr, Icechunk, or Kerchunk),
- Project fundraising (help write and administrate grants that will support VirtualiZarr),
- Improve documentation or tutorials,
- Attend the bi-weekly coordination meeting ([see Zarr community calendar](https://zarr.dev/community-calls/)),
- Contribute to the VirtualiZarr codebase.

(Matt Rocklin's post on [the role of a maintainer](https://matthewrocklin.com/blog/2019/05/18/maintainer) may be
interesting background reading, but should not be taken to strictly apply to the VirtualiZarr project.)

Obviously you are not expected to contribute in all (or even more than one) of these ways!
They are listed so as to indicate the many types of work that go into maintaining VirtualiZarr.

It is natural that your available time and enthusiasm for the project will wax and wane - this is fine and expected!
It is also common for core team members to have a "niche" - a particular part of the codebase they have specific expertise
with, or certain types of task above which they primarily perform.

If however you feel that is unlikely you will be able to be actively contribute in the foreseeable future
(or especially if you won't be available to answer questions about pieces of code that you wrote previously)
then you may want to consider letting us know you would rather be listed as an "Emeritus Core Team Member",
as this would help us in evaluating the overall health of the project.

## Issue triage

One of the main ways you might spend your contribution time is by responding to or triaging new issues.
Here’s a typical workflow for triaging a newly opened issue or discussion:

1. **Thank the reporter for opening an issue.**
    The issue tracker is many people’s first interaction with the VirtualiZarr project itself, beyond just using the library.
    It may also be their first open-source contribution of any kind. As such, we want it to be a welcoming, pleasant experience.
2. **Is the necessary information provided?**
    Ideally reporters would fill out the issue template, but many don’t. If crucial information (like the version of VirtualiZarr they used),
    is missing feel free to ask for that and label the issue with “needs info”.
    Make sure that the title accurately reflects the issue. Edit it yourself if it’s not clear.
    Remember also that issues can be converted to discussions and vice versa if appropriate.
3. **Is this a duplicate issue?**
    If a new issue is clearly a duplicate, label the new issue as “duplicate”, and close the issue with a link to the original issue.
    Make sure to still thank the reporter, and encourage them to chime in on the original issue, and perhaps try to fix it.
    If the new issue provides relevant information, such as a better or slightly different example, add it to the original issue as a comment or an edit to the original post.
4. **Is the issue minimal and reproducible?**
    For bug reports, we ask that the reporter provide a minimal reproducible example.
    See [minimal-bug-reports](https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports) for a good explanation.
    If the example is not reproducible, or if it’s clearly not minimal, feel free to ask the reporter if they can provide and example or simplify the provided one.
    Do acknowledge that writing minimal reproducible examples is hard work. If the reporter is struggling, you can try to write one yourself and we’ll edit the original post to include it.
    If a nice reproducible example has been provided, thank the reporter for that.
    If a reproducible example can’t be provided, add the “needs mcve” label.
    If a reproducible example is provided, but you see a simplification, edit the original post with your simpler reproducible example.
5. **Is this a clearly defined feature request?**
    Generally, VirtualiZarr prefers to discuss and design new features in issues, before a pull request is made.
    Encourage the submitter to include a proposed API for the new feature. Having them write a full docstring is a good way to pin down specifics.
6. **Is this a usage question?**
    We prefer that usage questions are asked as a [GitHub discussion topic](https://github.com/zarr-developers/VirtualiZarr/discussions).
    If it’s easy to answer, feel free to link to the relevant documentation section, let them know that in the future this kind of question should be on the discussions section, and close the issue.
7. **What labels and milestones should I add?**
    Apply the relevant labels. This is a bit of an art, and comes with experience. Look at similar issues to get a feel for how things are labeled.
    If the issue is clearly defined and the fix seems relatively straightforward, label the issue as `good-first-issue`.
8. **Where should the poster look to fix the issue?**
    If you can, it is very helpful to point to the approximate location in the codebase where a contributor might begin to fix the issue.
    This helps ease the way in for new contributors to the repository.

## Code review and contributions

As a core team member, you are a representative of the project,
and trusted to make decisions that will serve the long term interests
of all users. You also gain the responsibility of shepherding
other contributors through the review process; here are some
guidelines for how to do that.

### All contributors are treated the same

You should now have gained the ability to merge or approve
other contributors' pull requests.  Merging contributions is a shared power:
only merge contributions you yourself have carefully reviewed, and that are
clear improvements for the project. When in doubt, and especially for more
complex changes, wait until at least one other core team member has approved.
(See [Reviewing](#how-to-conduct-a-good-review) and especially
[Merge Only Changes You Understand](#merge-only-changes-you-understand) below.)

It should also be considered best practice to leave a reasonable (24hr) time window
after approval before merge to ensure that other core team members have a reasonable
chance to weigh in.

We are also an international community, with contributors from many different time zones,
some of whom will only contribute during their working hours, others who might only be able
to contribute during nights and weekends. It is important to be respectful of other peoples
schedules and working habits, even if it slows the project down slightly - we are in this
for the long run. In the same vein you also shouldn't feel pressured to be constantly
available or online, and users or contributors who are overly demanding and unreasonable
to the point of harassment will be directed to our [Code of Conduct](https://github.com/zarr-developers/virtualizarr?tab=coc-ov-file#readme).
We value sustainable development practices over mad rushes.

When merging, we automatically use GitHub's
[Squash and Merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/merging-a-pull-request#merging-a-pull-request)
to ensure a clean git history.

You should also continue to make your own pull requests as before and in accordance
with the [general contributing guide](https://virtualizarr.readthedocs.io/en/latest/contributing.html). These pull requests still
require the approval of another core team member before they can be merged.

### How to conduct a good review

*Always* be kind to contributors. Contributors are often doing
volunteer work, for which we are tremendously grateful. Provide
constructive criticism on ideas and implementations, and remind
yourself of how it felt when your own work was being evaluated as a
novice.

``VirtualiZarr`` strongly values mentorship in code review.  New users
often need more hand-holding, having little to no git
experience. Repeat yourself liberally, and, if you don’t recognize a
contributor, point them to our development guide, or other GitHub
workflow tutorials around the web. Do not assume that they know how
GitHub works (many don't realize that adding a commit
automatically updates a pull request, for example). Gentle, polite, kind
encouragement can make the difference between a new core team member and
an abandoned pull request.

### Merge only changes you understand

*Long-term maintainability* is an important concern.  Code doesn't
merely have to *work*, but should be *understood* by multiple core
developers.  Changes will have to be made in the future, and the
original contributor may have moved on.

Therefore, *do not merge a code change unless you understand it*. Ask
for help freely: we can consult community members, or even external developers,
for added insight where needed, and see this as a great learning opportunity.

While we collectively "own" any patches (and bugs!) that become part
of the code base, you are vouching for changes you merge.  Please take
that responsibility seriously.

Feel free to ping other active maintainers with any questions you may have.

## Further resources

As a core member, you should be familiar with community and developer
resources such as:

- Our [contributor guide](https://virtualizarr.readthedocs.io/en/stable/contributing.html).
- Our [code of conduct](https://github.com/zarr-developers/virtualizarr?tab=coc-ov-file#readme).
- [PEP8](https://peps.python.org/pep-0008/) for Python style.
- [PEP257](https://peps.python.org/pep-0257/) and the
   [NumPy documentation guide](https://numpy.org/devdocs/dev/howto-docs.html#documentation-style)
   for docstring conventions.
- [`pre-commit`](https://pre-commit.com) hooks for autoformatting.
- [`ruff`](https://github.com/astral-sh/ruff) autoformatting and linting.

You are not required to monitor any of the social resources.

Where possible we prefer to point people towards asynchronous forms of communication
like github issues instead of realtime chat options as they are far easier
for a global community to consume and refer back to.

We hold a [bi-weekly coordination meeting](https://zarr.dev/community-calls/) via video call.
This is a great place to bring up any questions you have, raise visibility of an issue and/or gather more perspectives.
Attendance is absolutely optional, and we keep the meeting to 1 hour in respect of your valuable time.
This meeting is public, so we occasionally have non-core team members join us.

## Inviting new core members

Any core member may nominate other contributors to join the core team.
While there is no hard-and-fast rule about who can be nominated, ideally,
they should have: been part of the project for at least two months, contributed
significant changes of their own, contributed to the discussion and
review of others' work, and collaborated in a way befitting our
community values. **We strongly encourage nominating anyone who has made significant non-code contributions
to the VirtualiZarr community in any way**. After nomination voting will happen on a private mailing list.
While it is expected that most votes will be unanimous, a two-thirds majority of
the cast votes is enough.

Core team members can choose to become emeritus core team members and suspend
their approval and voting rights until they become active again.

## Contribute to this guide (!)

This guide reflects the experience of the current core team members.  We
may well have missed things that, by now, have become second
nature—things that you, as a new team member, will spot more easily.
Please ask the other core team members if you have any questions, and
submit a pull request with insights gained.

## Conclusion

We are excited to have you on board!  We look forward to your
contributions to the code base and the community.  Thank you in
advance!
